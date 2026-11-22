"""Baseline evaluation pipeline for Speech Enhancement (Noisy, Spectral Gating, Demucs dns48)."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from pesq import pesq, PesqError
from pystoi import stoi
import torch

from src.models.demucs_loader import load_pretrained_dns48
from src.baselines.spectral_gating import SpectralGatingBaseline
from src.data.dataset import SpeechEnhancementDataset


def compute_utterance_metrics(
    clean_wav: np.ndarray,
    enhanced_wav: np.ndarray,
    sr: int = 16000,
) -> Tuple[Optional[float], Optional[float], str, str]:
    """Computes Wideband PESQ and STOI for an utterance pair."""
    # Ensure 1D float32 arrays
    c = np.asarray(clean_wav, dtype=np.float32).flatten()
    e = np.asarray(enhanced_wav, dtype=np.float32).flatten()

    min_len = min(len(c), len(e))
    if min_len == 0:
        return None, None, "Zero length audio array", "Zero length audio array"

    c = c[:min_len]
    e = e[:min_len]

    if not np.isfinite(c).all():
        return None, None, "Clean audio contains non-finite values", "Clean audio contains non-finite values"

    if not np.isfinite(e).all():
        e = np.nan_to_num(e, nan=0.0, posinf=1.0, neginf=-1.0)

    e = np.clip(e, -1.0, 1.0)

    # 1. PESQ (Wideband)
    pesq_val = None
    pesq_err = ""
    try:
        score = pesq(sr, c, e, "wb")
        pesq_val = round(float(score), 4)
    except (PesqError, Exception) as err:
        pesq_err = str(err)

    # 2. STOI (standard)
    stoi_val = None
    stoi_err = ""
    try:
        score = stoi(c, e, sr, extended=False)
        stoi_val = round(float(score), 4)
    except Exception as err:
        stoi_err = str(err)

    return pesq_val, stoi_val, pesq_err, stoi_err


def _eval_single_utterance_worker(args: Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Worker function for parallel baseline enhancement and metric computation."""
    meta, clean_np, noisy_np, dns48_np, sg_params = args
    sg_baseline = SpectralGatingBaseline(**sg_params)

    results = []

    # 1. Noisy (unprocessed)
    p_noisy, s_noisy, pe_noisy, se_noisy = compute_utterance_metrics(clean_np, noisy_np, sr=sg_params.get("sr", 16000))
    results.append({
        "utterance_id": meta["utterance_id"],
        "speaker_id": meta["speaker_id"],
        "noise_type": meta["noise_type"],
        "snr_db": meta["snr_db"],
        "duration_sec": meta["duration_sec"],
        "condition": "noisy",
        "pesq_wb": p_noisy if p_noisy is not None else "",
        "stoi": s_noisy if s_noisy is not None else "",
        "pesq_error": pe_noisy,
        "stoi_error": se_noisy,
    })

    # 2. Spectral Gating
    enhanced_sg = sg_baseline.enhance(noisy_np)
    p_sg, s_sg, pe_sg, se_sg = compute_utterance_metrics(clean_np, enhanced_sg, sr=sg_params.get("sr", 16000))
    results.append({
        "utterance_id": meta["utterance_id"],
        "speaker_id": meta["speaker_id"],
        "noise_type": meta["noise_type"],
        "snr_db": meta["snr_db"],
        "duration_sec": meta["duration_sec"],
        "condition": "spectral_gating",
        "pesq_wb": p_sg if p_sg is not None else "",
        "stoi": s_sg if s_sg is not None else "",
        "pesq_error": pe_sg,
        "stoi_error": se_sg,
    })

    # 3. Pretrained Demucs dns48
    p_demucs, s_demucs, pe_demucs, se_demucs = compute_utterance_metrics(clean_np, dns48_np, sr=sg_params.get("sr", 16000))
    results.append({
        "utterance_id": meta["utterance_id"],
        "speaker_id": meta["speaker_id"],
        "noise_type": meta["noise_type"],
        "snr_db": meta["snr_db"],
        "duration_sec": meta["duration_sec"],
        "condition": "dns48_zeroshot",
        "pesq_wb": p_demucs if p_demucs is not None else "",
        "stoi": s_demucs if s_demucs is not None else "",
        "pesq_error": pe_demucs,
        "stoi_error": se_demucs,
    })

    return results


def calculate_group_statistics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Mean, Median, Std, Min, Max, and counts for PESQ and STOI."""
    pesq_vals = [float(r["pesq_wb"]) for r in rows if r.get("pesq_wb") != ""]
    stoi_vals = [float(r["stoi"]) for r in rows if r.get("stoi") != ""]

    pesq_stats = {
        "count_successful": len(pesq_vals),
        "count_failed": len(rows) - len(pesq_vals),
        "mean": round(float(np.mean(pesq_vals)), 4) if pesq_vals else None,
        "std": round(float(np.std(pesq_vals)), 4) if pesq_vals else None,
        "median": round(float(np.median(pesq_vals)), 4) if pesq_vals else None,
        "min": round(float(np.min(pesq_vals)), 4) if pesq_vals else None,
        "max": round(float(np.max(pesq_vals)), 4) if pesq_vals else None,
    }

    stoi_stats = {
        "count_successful": len(stoi_vals),
        "count_failed": len(rows) - len(stoi_vals),
        "mean": round(float(np.mean(stoi_vals)), 4) if stoi_vals else None,
        "std": round(float(np.std(stoi_vals)), 4) if stoi_vals else None,
        "median": round(float(np.median(stoi_vals)), 4) if stoi_vals else None,
        "min": round(float(np.min(stoi_vals)), 4) if stoi_vals else None,
        "max": round(float(np.max(stoi_vals)), 4) if stoi_vals else None,
    }

    return {
        "num_utterances": len(rows),
        "pesq_wb": pesq_stats,
        "stoi": stoi_stats,
    }


class BaselineEvaluator:
    """Manages baseline speech enhancement evaluation on validation and test sets."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda",
        max_workers: Optional[int] = None,
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.max_workers = max_workers or min(os.cpu_count() or 4, 8)

        base_cfg = self.config.get("baseline", {})
        self.sg_params = {
            "sr": base_cfg.get("sr", 16000),
            "stationary": base_cfg.get("stationary", True),
            "n_fft": base_cfg.get("n_fft", 512),
            "win_length": base_cfg.get("win_length", 512),
            "hop_length": base_cfg.get("hop_length", 128),
            "n_std_thresh_stationary": base_cfg.get("n_std_thresh_stationary", 1.5),
            "prop_decrease": base_cfg.get("prop_decrease", 1.0),
        }

        print(f"Loading pretrained Demucs dns48 on {self.device}...")
        self.model = load_pretrained_dns48(device=str(self.device))
        self.model.eval()

    def evaluate_split(
        self,
        manifest_path: Union[str, Path],
        local_data_dir: Union[str, Path] = "/content/data_16k",
        split_name: str = "val",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluates all utterances in a split across Noisy, Spectral Gating, and dns48."""
        dataset = SpeechEnhancementDataset(
            manifest_path=manifest_path,
            local_data_dir=local_data_dir,
            split=split_name,
            sample_rate=self.sg_params["sr"],
        )

        total_items = len(dataset)
        print(f"\nStarting evaluation of split '{split_name}' ({total_items} utterances)...")

        # 1. Run dns48 GPU inference for all utterances
        tasks = []
        inference_start = time.time()
        print(f"Running dns48 inference on {self.device}...")

        with torch.no_grad():
            for i in range(total_items):
                item = dataset[i]
                noisy_wav = item["noisy"].unsqueeze(0).to(self.device)  # [1, 1, T]
                dns48_wav = self.model(noisy_wav)  # [1, 1, T]

                clean_np = item["clean"].squeeze(0).numpy()
                noisy_np = item["noisy"].squeeze(0).numpy()
                dns48_np = dns48_wav.squeeze(0).squeeze(0).cpu().numpy()

                meta = {
                    "utterance_id": item["utterance_id"],
                    "speaker_id": item["speaker_id"],
                    "noise_type": item["noise_type"],
                    "snr_db": item["snr_db"],
                    "duration_sec": item["duration_sec"],
                }
                tasks.append((meta, clean_np, noisy_np, dns48_np, self.sg_params))

        inference_time = time.time() - inference_start
        print(f"Completed dns48 inference for {total_items} utterances in {inference_time:.2f}s "
              f"({total_items / inference_time:.1f} utts/sec)")

        # 2. Parallel evaluation of Spectral Gating and Metrics
        print(f"Computing metrics with {self.max_workers} CPU worker processes...")
        all_rows: List[Dict[str, Any]] = []
        eval_start = time.time()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_eval_single_utterance_worker, t) for t in tasks]
            completed_count = 0
            for fut in as_completed(futures):
                rows = fut.result()
                all_rows.extend(rows)
                completed_count += 1
                if completed_count % 100 == 0 or completed_count == total_items:
                    print(f"  Evaluated {completed_count}/{total_items} utterances "
                          f"({completed_count / (time.time() - eval_start):.1f} utts/sec)")

        total_eval_time = time.time() - eval_start
        print(f"Metrics computation completed in {total_eval_time:.2f}s.")

        # Sort rows deterministically by utterance_id, condition
        all_rows.sort(key=lambda r: (r["utterance_id"], r["condition"]))

        # 3. Generate Hierarchical Summaries
        conditions = ["noisy", "spectral_gating", "dns48_zeroshot"]
        summary: Dict[str, Any] = {
            "split": split_name,
            "num_utterances": total_items,
            "conditions": {},
            "breakdowns": {},
        }

        # Per-condition overall statistics
        for cond in conditions:
            cond_rows = [r for r in all_rows if r["condition"] == cond]
            summary["conditions"][cond] = calculate_group_statistics(cond_rows)

        # Breakdown by SNR level, Noise Type, and Speaker
        snr_levels = sorted(set(r["snr_db"] for r in all_rows))
        noise_types = sorted(set(r["noise_type"] for r in all_rows))
        speakers = sorted(set(r["speaker_id"] for r in all_rows))

        summary["breakdowns"]["by_snr"] = {}
        for snr in snr_levels:
            summary["breakdowns"]["by_snr"][str(snr)] = {
                cond: calculate_group_statistics([r for r in all_rows if r["condition"] == cond and r["snr_db"] == snr])
                for cond in conditions
            }

        summary["breakdowns"]["by_noise"] = {}
        for n in noise_types:
            summary["breakdowns"]["by_noise"][n] = {
                cond: calculate_group_statistics([r for r in all_rows if r["condition"] == cond and r["noise_type"] == n])
                for cond in conditions
            }

        summary["breakdowns"]["by_speaker"] = {}
        for spk in speakers:
            summary["breakdowns"]["by_speaker"][spk] = {
                cond: calculate_group_statistics([r for r in all_rows if r["condition"] == cond and r["speaker_id"] == spk])
                for cond in conditions
            }

        return all_rows, summary

    def save_results(
        self,
        all_rows: List[Dict[str, Any]],
        summary: Dict[str, Any],
        output_prefix: Union[str, Path],
    ) -> None:
        """Saves detailed per-utterance CSV, structured JSON summary, and flat CSV summary."""
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)

        per_utt_csv = output_prefix.parent / f"{output_prefix.name}_per_utterance.csv"
        summary_json = output_prefix.parent / f"{output_prefix.name}_summary.json"
        summary_csv = output_prefix.parent / f"{output_prefix.name}_summary.csv"

        # 1. Save per-utterance CSV
        if all_rows:
            fieldnames = list(all_rows[0].keys())
            with open(per_utt_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"Saved per-utterance results to {per_utt_csv}")

        # 2. Save summary JSON
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary JSON to {summary_json}")

        # 3. Save flat summary CSV for easy reporting/tables
        flat_rows = []
        # Overall
        for cond, stats in summary["conditions"].items():
            flat_rows.append({
                "group_type": "overall",
                "group_value": "all",
                "condition": cond,
                "num_utterances": stats["num_utterances"],
                "pesq_mean": stats["pesq_wb"]["mean"],
                "pesq_median": stats["pesq_wb"]["median"],
                "pesq_std": stats["pesq_wb"]["std"],
                "pesq_min": stats["pesq_wb"]["min"],
                "pesq_max": stats["pesq_wb"]["max"],
                "pesq_success": stats["pesq_wb"]["count_successful"],
                "pesq_failed": stats["pesq_wb"]["count_failed"],
                "stoi_mean": stats["stoi"]["mean"],
                "stoi_median": stats["stoi"]["median"],
                "stoi_std": stats["stoi"]["std"],
                "stoi_min": stats["stoi"]["min"],
                "stoi_max": stats["stoi"]["max"],
                "stoi_success": stats["stoi"]["count_successful"],
                "stoi_failed": stats["stoi"]["count_failed"],
            })

        # By SNR
        for snr, cond_dict in summary["breakdowns"]["by_snr"].items():
            for cond, stats in cond_dict.items():
                flat_rows.append({
                    "group_type": "snr_db",
                    "group_value": snr,
                    "condition": cond,
                    "num_utterances": stats["num_utterances"],
                    "pesq_mean": stats["pesq_wb"]["mean"],
                    "pesq_median": stats["pesq_wb"]["median"],
                    "pesq_std": stats["pesq_wb"]["std"],
                    "pesq_min": stats["pesq_wb"]["min"],
                    "pesq_max": stats["pesq_wb"]["max"],
                    "pesq_success": stats["pesq_wb"]["count_successful"],
                    "pesq_failed": stats["pesq_wb"]["count_failed"],
                    "stoi_mean": stats["stoi"]["mean"],
                    "stoi_median": stats["stoi"]["median"],
                    "stoi_std": stats["stoi"]["std"],
                    "stoi_min": stats["stoi"]["min"],
                    "stoi_max": stats["stoi"]["max"],
                    "stoi_success": stats["stoi"]["count_successful"],
                    "stoi_failed": stats["stoi"]["count_failed"],
                })

        # By Noise
        for noise, cond_dict in summary["breakdowns"]["by_noise"].items():
            for cond, stats in cond_dict.items():
                flat_rows.append({
                    "group_type": "noise_type",
                    "group_value": noise,
                    "condition": cond,
                    "num_utterances": stats["num_utterances"],
                    "pesq_mean": stats["pesq_wb"]["mean"],
                    "pesq_median": stats["pesq_wb"]["median"],
                    "pesq_std": stats["pesq_wb"]["std"],
                    "pesq_min": stats["pesq_wb"]["min"],
                    "pesq_max": stats["pesq_wb"]["max"],
                    "pesq_success": stats["pesq_wb"]["count_successful"],
                    "pesq_failed": stats["pesq_wb"]["count_failed"],
                    "stoi_mean": stats["stoi"]["mean"],
                    "stoi_median": stats["stoi"]["median"],
                    "stoi_std": stats["stoi"]["std"],
                    "stoi_min": stats["stoi"]["min"],
                    "stoi_max": stats["stoi"]["max"],
                    "stoi_success": stats["stoi"]["count_successful"],
                    "stoi_failed": stats["stoi"]["count_failed"],
                })

        # By Speaker
        for spk, cond_dict in summary["breakdowns"]["by_speaker"].items():
            for cond, stats in cond_dict.items():
                flat_rows.append({
                    "group_type": "speaker_id",
                    "group_value": spk,
                    "condition": cond,
                    "num_utterances": stats["num_utterances"],
                    "pesq_mean": stats["pesq_wb"]["mean"],
                    "pesq_median": stats["pesq_wb"]["median"],
                    "pesq_std": stats["pesq_wb"]["std"],
                    "pesq_min": stats["pesq_wb"]["min"],
                    "pesq_max": stats["pesq_wb"]["max"],
                    "pesq_success": stats["pesq_wb"]["count_successful"],
                    "pesq_failed": stats["pesq_wb"]["count_failed"],
                    "stoi_mean": stats["stoi"]["mean"],
                    "stoi_median": stats["stoi"]["median"],
                    "stoi_std": stats["stoi"]["std"],
                    "stoi_min": stats["stoi"]["min"],
                    "stoi_max": stats["stoi"]["max"],
                    "stoi_success": stats["stoi"]["count_successful"],
                    "stoi_failed": stats["stoi"]["count_failed"],
                })

        if flat_rows:
            fieldnames = list(flat_rows[0].keys())
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_rows)
            print(f"Saved summary CSV to {summary_csv}")
