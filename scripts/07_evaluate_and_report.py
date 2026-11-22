"""Script 07: Official Test-Set Evaluation and Comparative Analysis (Phase 4).

Evaluates the fine-tuned Demucs dns48 model (best_val_pesq.th, selected at Epoch 14)
on the official Edinburgh VoiceBank-DEMAND test set (824 utterances, speakers p232 & p257).
Combines historical Phase 3A baselines (Noisy, Spectral Gating, Demucs dns48 Zero-Shot)
with the fine-tuned model for unified 4-way comparative analysis, group breakdowns,
publication figures, and representative audio exports.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import torch
import torch.nn as nn
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.spectral_gating import SpectralGatingBaseline
from src.data.dataset import SpeechEnhancementDataset, ensure_local_data_16k
from src.evaluation.evaluator import compute_utterance_metrics, calculate_group_statistics
from src.models.demucs_loader import (
    build_demucs_dns48_architecture,
    load_pretrained_dns48,
    load_checkpoint,
    verify_demucs_dns48_attributes,
)


def save_wav_16k_pcm16(path: Union[str, Path], audio_np: np.ndarray, sr: int = 16000) -> None:
    """Saves a 1D float32 audio array as 16 kHz 16-bit PCM mono WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio_np, dtype=np.float32).flatten()
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, audio_int16)


def _finetuned_eval_worker(args: Tuple[Dict[str, Any], np.ndarray, np.ndarray, int]) -> Dict[str, Any]:
    """Worker function for parallel PESQ and STOI computation of fine-tuned model output."""
    meta, clean_np, enh_np, sr = args
    pesq_val, stoi_val, pesq_err, stoi_err = compute_utterance_metrics(clean_np, enh_np, sr=sr)
    success = pesq_val is not None and stoi_val is not None
    err_msg = ""
    if pesq_err:
        err_msg += f"PESQ: {pesq_err}; "
    if stoi_err:
        err_msg += f"STOI: {stoi_err}"

    return {
        "utterance_id": meta["utterance_id"],
        "speaker_id": meta["speaker_id"],
        "noise_type": meta["noise_type"],
        "snr_db": meta["snr_db"],
        "duration_sec": meta["duration_sec"],
        "num_samples": meta["num_samples"],
        "condition": "dns48_finetuned",
        "pesq_wb": pesq_val if pesq_val is not None else "",
        "stoi": stoi_val if stoi_val is not None else "",
        "success": success,
        "error": err_msg.strip(),
    }


class OfficialTestEvaluator:
    """Orchestrates Phase 4 official test-set evaluation and 4-way comparative analysis."""

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: Union[str, Path],
        test_manifest_path: Union[str, Path] = "manifests/test_manifest.json",
        baseline_csv_path: Union[str, Path] = "results/test_baseline_per_utterance.csv",
        output_dir: Union[str, Path] = "results",
        device: Optional[str] = None,
        max_workers: Optional[int] = None,
    ):
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.test_manifest_path = Path(test_manifest_path)
        self.baseline_csv_path = Path(baseline_csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers or min(os.cpu_count() or 4, 8)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[INIT] OfficialTestEvaluator configured on device: {self.device}")
        print(f"[INIT] Checkpoint path: {self.checkpoint_path}")
        print(f"[INIT] Test manifest: {self.test_manifest_path}")
        print(f"[INIT] Baseline CSV: {self.baseline_csv_path}")
        print(f"[INIT] Output directory: {self.output_dir}")

    def run_preflight_check(self) -> Dict[str, Any]:
        """Executes mandatory pre-evaluation assertions and a 5-sample mini test."""
        print("\n" + "=" * 70)
        print("STAGE 1: CONTROLLED PRE-EVALUATION PREFLIGHT CHECK")
        print("=" * 70)

        # 1. Verify Checkpoint File & Architecture
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

        file_size = self.checkpoint_path.stat().st_size
        print(f"[CHECK 1/3] Loading checkpoint: {self.checkpoint_path} ({file_size:,} bytes)...")
        model = build_demucs_dns48_architecture()
        payload = load_checkpoint(self.checkpoint_path, model=model, device=self.device)
        verify_demucs_dns48_attributes(model)
        model.eval()

        epoch = payload.get("epoch", None)
        model_sd = payload.get("model_state_dict", {})
        print(f"  Checkpoint Epoch: {epoch} (Expected: 14)")
        print(f"  State Dict Keys: {len(model_sd)} (Expected: 48)")
        assert epoch == 14, f"Expected Epoch 14 checkpoint, got {epoch}"
        assert len(model_sd) == 48, f"Expected 48 state dict keys, got {len(model_sd)}"
        print("  [PASSED] Checkpoint integrity and Demucs dns48 architecture verified.")

        # 2. Verify Test Dataset and Baseline Data
        print("[CHECK 2/3] Verifying test dataset and Phase 3A baseline records...")
        local_dir = self.config.get("paths", {}).get("local_data_16k_dir", "/content/data_16k")
        tar_path = self.config.get("paths", {}).get("tar_16k_path", "/content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar")
        ensure_local_data_16k(local_dir, tar_path=tar_path)
        dataset = SpeechEnhancementDataset(
            manifest_path=self.test_manifest_path,
            local_data_dir=local_dir,
            split="test",
            sample_rate=16000,
        )
        assert len(dataset) == 824, f"Expected 824 test utterances, got {len(dataset)}"
        speakers = {item["speaker_id"] for item in dataset.manifest}
        assert speakers == {"p232", "p257"}, f"Expected speakers {{'p232', 'p257'}}, got {speakers}"

        p232_count = sum(1 for item in dataset.manifest if item["speaker_id"] == "p232")
        p257_count = sum(1 for item in dataset.manifest if item["speaker_id"] == "p257")
        assert p232_count == 393, f"Expected 393 p232 utterances, got {p232_count}"
        assert p257_count == 431, f"Expected 431 p257 utterances, got {p257_count}"

        if not self.baseline_csv_path.exists():
            raise FileNotFoundError(f"Historical Phase 3A baseline CSV not found: {self.baseline_csv_path}")

        with open(self.baseline_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            baseline_rows = list(reader)
        assert len(baseline_rows) == 2472, f"Expected 2472 baseline rows (824x3), got {len(baseline_rows)}"
        baseline_utt_ids = {r["utterance_id"] for r in baseline_rows}
        test_utt_ids = {item["utterance_id"] for item in dataset.manifest}
        assert baseline_utt_ids == test_utt_ids, "Utterance ID mismatch between test manifest and baseline CSV"
        print("  [PASSED] Dataset (824 utts, p232: 393, p257: 431) and Baseline CSV (2472 rows) verified.")

        # 3. 5-Sample Mini Evaluation Test
        print("[CHECK 3/3] Running 5-sample mini evaluation test...")
        mini_results = []
        with torch.no_grad():
            for i in range(5):
                item = dataset[i]
                noisy_wav = item["noisy"].unsqueeze(0).to(self.device)
                enh_wav = model(noisy_wav)
                clean_np = item["clean"].squeeze(0).numpy()
                enh_np = enh_wav.squeeze(0).squeeze(0).cpu().numpy()

                assert enh_np.shape == clean_np.shape, f"Length mismatch: {enh_np.shape} vs {clean_np.shape}"
                assert np.isfinite(enh_np).all(), "Enhanced audio contains non-finite values"

                p_val, s_val, p_err, s_err = compute_utterance_metrics(clean_np, enh_np, sr=16000)
                assert p_val is not None, f"PESQ calculation failed on item {i}: {p_err}"
                assert s_val is not None, f"STOI calculation failed on item {i}: {s_err}"
                assert 1.0 <= p_val <= 4.5, f"PESQ value {p_val} out of expected range"
                assert 0.0 <= s_val <= 1.0, f"STOI value {s_val} out of expected range"

                mini_results.append({
                    "utterance_id": item["utterance_id"],
                    "speaker": item["speaker_id"],
                    "noise": item["noise_type"],
                    "snr": item["snr_db"],
                    "pesq": p_val,
                    "stoi": s_val,
                })
                print(f"    Item {i+1}/5 [{item['utterance_id']}]: PESQ={p_val:.4f}, STOI={s_val:.4f}")

        print("  [PASSED] 5-sample mini evaluation succeeded with valid metrics.")
        print("=" * 70)
        print("ALL PREFLIGHT CHECKS PASSED — AUTHORIZED FOR FULL EVALUATION\n")

        return {
            "checkpoint_verified": True,
            "epoch": epoch,
            "dataset_utterances": len(dataset),
            "speakers": {"p232": p232_count, "p257": p257_count},
            "mini_results": mini_results,
        }

    def evaluate_test_set(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Performs full 824-utterance GPU inference and parallel metric computation."""
        print("=" * 70)
        print("STAGE 2: FULL 824-UTTERANCE EVALUATION (FINE-TUNED DEMUCS DNS48)")
        print("=" * 70)

        # 1. Load fine-tuned model
        model = build_demucs_dns48_architecture()
        load_checkpoint(self.checkpoint_path, model=model, device=self.device)
        model.eval()

        # 2. Load test dataset
        dataset = SpeechEnhancementDataset(
            manifest_path=self.test_manifest_path,
            local_data_dir=self.config.get("paths", {}).get("local_data_16k_dir", "/content/data_16k"),
            split="test",
            sample_rate=16000,
        )
        total_items = len(dataset)

        # 3. GPU Inference
        print(f"Running fine-tuned Demucs inference on {self.device} for all {total_items} utterances...")
        infer_start = time.time()
        tasks = []
        enhanced_cache: Dict[str, np.ndarray] = {}

        with torch.no_grad():
            for i in range(total_items):
                item = dataset[i]
                noisy_wav = item["noisy"].unsqueeze(0).to(self.device)
                enh_wav = model(noisy_wav)

                clean_np = item["clean"].squeeze(0).numpy()
                noisy_np = item["noisy"].squeeze(0).numpy()
                enh_np = enh_wav.squeeze(0).squeeze(0).cpu().numpy()

                meta = {
                    "utterance_id": item["utterance_id"],
                    "speaker_id": item["speaker_id"],
                    "noise_type": item["noise_type"],
                    "snr_db": item["snr_db"],
                    "duration_sec": item["duration_sec"],
                    "num_samples": item["num_samples"],
                }
                enhanced_cache[item["utterance_id"]] = enh_np
                tasks.append((meta, clean_np, enh_np, 16000))

        infer_duration = time.time() - infer_start
        print(f"Inference complete: {total_items} utterances in {infer_duration:.2f}s "
              f"({total_items / infer_duration:.1f} utts/sec).")

        # 4. Multi-core Metric Calculation
        print(f"Computing Wideband PESQ and STOI across {self.max_workers} CPU worker processes...")
        eval_start = time.time()
        finetuned_rows: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_finetuned_eval_worker, t) for t in tasks]
            completed = 0
            for fut in as_completed(futures):
                res = fut.result()
                finetuned_rows.append(res)
                completed += 1
                if completed % 200 == 0 or completed == total_items:
                    print(f"  Processed {completed}/{total_items} utterances "
                          f"({completed / (time.time() - eval_start):.1f} utts/sec)...")

        eval_duration = time.time() - eval_start
        print(f"Metric computation complete in {eval_duration:.2f}s.")

        # Sort deterministically by utterance_id
        finetuned_rows.sort(key=lambda r: r["utterance_id"])

        # 5. Save per-utterance fine-tuned results CSV
        ft_csv_path = self.output_dir / "test_finetuned_per_utterance.csv"
        print(f"Saving per-utterance results to {ft_csv_path}...")
        fieldnames = [
            "utterance_id", "speaker_id", "noise_type", "snr_db", "duration_sec",
            "num_samples", "condition", "pesq_wb", "stoi", "success", "error"
        ]
        with open(ft_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(finetuned_rows)

        # 6. Compute fine-tuned summary
        ft_stats = calculate_group_statistics(finetuned_rows)
        ft_summary = {
            "condition": "dns48_finetuned",
            "checkpoint": str(self.checkpoint_path),
            "total_utterances": total_items,
            "inference_duration_sec": round(infer_duration, 2),
            "eval_duration_sec": round(eval_duration, 2),
            "stats": ft_stats,
        }
        ft_summary_path = self.output_dir / "test_finetuned_summary.json"
        with open(ft_summary_path, "w", encoding="utf-8") as f:
            json.dump(ft_summary, f, indent=2)
        print(f"Saved fine-tuned summary to {ft_summary_path}.")

        return finetuned_rows, enhanced_cache

    def compile_comparative_analysis(
        self,
        finetuned_rows: List[Dict[str, Any]],
        enhanced_cache: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Merges historical Phase 3A baselines with fine-tuned model for unified 4-way analysis."""
        print("\n" + "=" * 70)
        print("STAGE 3: UNIFIED 4-WAY COMPARATIVE ANALYSIS & BREAKDOWNS")
        print("=" * 70)

        # 1. Load historical Phase 3A baseline rows
        with open(self.baseline_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            baseline_rows = list(reader)

        # Organize rows by condition
        conditions = ["noisy", "spectral_gating", "dns48_zeroshot", "dns48_finetuned"]
        condition_rows: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}

        for r in baseline_rows:
            c = r["condition"]
            if c in condition_rows:
                condition_rows[c].append(r)

        for r in finetuned_rows:
            condition_rows["dns48_finetuned"].append(r)

        for c in conditions:
            assert len(condition_rows[c]) == 824, f"Condition {c} has {len(condition_rows[c])} rows, expected 824"

        # Organize by utterance_id for paired comparisons
        by_utt: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for c in conditions:
            for r in condition_rows[c]:
                uid = r["utterance_id"]
                if uid not in by_utt:
                    by_utt[uid] = {}
                by_utt[uid][c] = r

        # 2. Overall Summary Statistics Table
        print("\n--- OVERALL 4-WAY COMPARATIVE SUMMARY ---")
        summary_table: List[Dict[str, Any]] = []
        overall_stats: Dict[str, Any] = {}

        for c in conditions:
            rows = condition_rows[c]
            pesq_vals = [float(r["pesq_wb"]) for r in rows if r.get("pesq_wb") != ""]
            stoi_vals = [float(r["stoi"]) for r in rows if r.get("stoi") != ""]
            stats = {
                "condition": c,
                "num_utterances": len(rows),
                "successful": len(pesq_vals),
                "failed": len(rows) - len(pesq_vals),
                "pesq_mean": round(float(np.mean(pesq_vals)), 4),
                "pesq_std": round(float(np.std(pesq_vals)), 4),
                "pesq_median": round(float(np.median(pesq_vals)), 4),
                "pesq_min": round(float(np.min(pesq_vals)), 4),
                "pesq_max": round(float(np.max(pesq_vals)), 4),
                "stoi_mean": round(float(np.mean(stoi_vals)), 4),
                "stoi_std": round(float(np.std(stoi_vals)), 4),
                "stoi_median": round(float(np.median(stoi_vals)), 4),
                "stoi_min": round(float(np.min(stoi_vals)), 4),
                "stoi_max": round(float(np.max(stoi_vals)), 4),
            }
            overall_stats[c] = stats

        # Compute Deltas vs Reference Conditions
        ref_noisy = overall_stats["noisy"]
        ref_zs = overall_stats["dns48_zeroshot"]
        ref_sg = overall_stats["spectral_gating"]
        ft_stats = overall_stats["dns48_finetuned"]

        for c in conditions:
            s = overall_stats[c]
            # Deltas vs Noisy
            delta_p_noisy = round(s["pesq_mean"] - ref_noisy["pesq_mean"], 4)
            rel_p_noisy = round((s["pesq_mean"] - ref_noisy["pesq_mean"]) / ref_noisy["pesq_mean"] * 100.0, 2)
            delta_s_noisy = round(s["stoi_mean"] - ref_noisy["stoi_mean"], 4)
            rel_s_noisy = round((s["stoi_mean"] - ref_noisy["stoi_mean"]) / ref_noisy["stoi_mean"] * 100.0, 2)

            # Deltas vs Zero-Shot
            delta_p_zs = round(s["pesq_mean"] - ref_zs["pesq_mean"], 4)
            rel_p_zs = round((s["pesq_mean"] - ref_zs["pesq_mean"]) / ref_zs["pesq_mean"] * 100.0, 2)
            delta_s_zs = round(s["stoi_mean"] - ref_zs["stoi_mean"], 4)
            rel_s_zs = round((s["stoi_mean"] - ref_zs["stoi_mean"]) / ref_zs["stoi_mean"] * 100.0, 2)

            row_summary = {
                "condition": c,
                "num_utterances": s["num_utterances"],
                "pesq_mean": s["pesq_mean"],
                "pesq_std": s["pesq_std"],
                "pesq_median": s["pesq_median"],
                "pesq_min": s["pesq_min"],
                "pesq_max": s["pesq_max"],
                "stoi_mean": s["stoi_mean"],
                "stoi_std": s["stoi_std"],
                "stoi_median": s["stoi_median"],
                "stoi_min": s["stoi_min"],
                "stoi_max": s["stoi_max"],
                "delta_pesq_vs_noisy": delta_p_noisy,
                "rel_pesq_vs_noisy_pct": rel_p_noisy,
                "delta_stoi_vs_noisy": delta_s_noisy,
                "rel_stoi_vs_noisy_pct": rel_s_noisy,
                "delta_pesq_vs_zeroshot": delta_p_zs,
                "rel_pesq_vs_zeroshot_pct": rel_p_zs,
                "delta_stoi_vs_zeroshot": delta_s_zs,
                "rel_stoi_vs_zeroshot_pct": rel_s_zs,
            }
            summary_table.append(row_summary)
            print(f"  {c:18s} | PESQ: {s['pesq_mean']:.4f} ± {s['pesq_std']:.4f} (med: {s['pesq_median']:.4f}) | "
                  f"STOI: {s['stoi_mean']:.4f} ± {s['stoi_std']:.4f} (med: {s['stoi_median']:.4f}) | "
                  f"dPESQ(noisy): {delta_p_noisy:+.4f} | dPESQ(zs): {delta_p_zs:+.4f}")

        # Save final_test_comparison.csv
        comparison_csv_path = self.output_dir / "final_test_comparison.csv"
        with open(comparison_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_table[0].keys()))
            writer.writeheader()
            writer.writerows(summary_table)
        print(f"Saved overall comparison table to {comparison_csv_path}.")

        # 3. Breakdown by Speaker (p232: 393, p257: 431)
        print("\n--- BREAKDOWN BY SPEAKER ---")
        speaker_table = []
        for spk in ["p232", "p257"]:
            spk_noisy = np.mean([float(r["pesq_wb"]) for r in condition_rows["noisy"] if r["speaker_id"] == spk])
            spk_zs = np.mean([float(r["pesq_wb"]) for r in condition_rows["dns48_zeroshot"] if r["speaker_id"] == spk])
            spk_noisy_s = np.mean([float(r["stoi"]) for r in condition_rows["noisy"] if r["speaker_id"] == spk])
            spk_zs_s = np.mean([float(r["stoi"]) for r in condition_rows["dns48_zeroshot"] if r["speaker_id"] == spk])

            for c in conditions:
                rows = [r for r in condition_rows[c] if r["speaker_id"] == spk]
                p_vals = [float(r["pesq_wb"]) for r in rows]
                s_vals = [float(r["stoi"]) for r in rows]
                p_mean = float(np.mean(p_vals))
                s_mean = float(np.mean(s_vals))
                row = {
                    "speaker_id": spk,
                    "condition": c,
                    "num_utterances": len(rows),
                    "pesq_mean": round(p_mean, 4),
                    "pesq_std": round(float(np.std(p_vals)), 4),
                    "pesq_median": round(float(np.median(p_vals)), 4),
                    "stoi_mean": round(s_mean, 4),
                    "stoi_std": round(float(np.std(s_vals)), 4),
                    "stoi_median": round(float(np.median(s_vals)), 4),
                    "delta_pesq_vs_noisy": round(p_mean - spk_noisy, 4),
                    "delta_pesq_vs_zeroshot": round(p_mean - spk_zs, 4),
                    "delta_stoi_vs_noisy": round(s_mean - spk_noisy_s, 4),
                    "delta_stoi_vs_zeroshot": round(s_mean - spk_zs_s, 4),
                }
                speaker_table.append(row)
                print(f"  [{spk}] {c:18s} | PESQ: {p_mean:.4f} | STOI: {s_mean:.4f} | dPESQ(zs): {row['delta_pesq_vs_zeroshot']:+.4f}")

        spk_csv_path = self.output_dir / "test_breakdown_by_speaker.csv"
        with open(spk_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(speaker_table[0].keys()))
            writer.writeheader()
            writer.writerows(speaker_table)
        print(f"Saved speaker breakdown to {spk_csv_path}.")

        # 4. Breakdown by Noise Type (5 unseen noises: bus, cafe, living, office, psquare)
        print("\n--- BREAKDOWN BY NOISE TYPE ---")
        noise_table = []
        noise_types = ["bus", "cafe", "living", "office", "psquare"]
        for noise in noise_types:
            n_noisy = np.mean([float(r["pesq_wb"]) for r in condition_rows["noisy"] if r["noise_type"] == noise])
            n_zs = np.mean([float(r["pesq_wb"]) for r in condition_rows["dns48_zeroshot"] if r["noise_type"] == noise])
            n_noisy_s = np.mean([float(r["stoi"]) for r in condition_rows["noisy"] if r["noise_type"] == noise])
            n_zs_s = np.mean([float(r["stoi"]) for r in condition_rows["dns48_zeroshot"] if r["noise_type"] == noise])

            for c in conditions:
                rows = [r for r in condition_rows[c] if r["noise_type"] == noise]
                p_vals = [float(r["pesq_wb"]) for r in rows]
                s_vals = [float(r["stoi"]) for r in rows]
                p_mean = float(np.mean(p_vals))
                s_mean = float(np.mean(s_vals))
                row = {
                    "noise_type": noise,
                    "condition": c,
                    "num_utterances": len(rows),
                    "pesq_mean": round(p_mean, 4),
                    "pesq_std": round(float(np.std(p_vals)), 4),
                    "pesq_median": round(float(np.median(p_vals)), 4),
                    "stoi_mean": round(s_mean, 4),
                    "stoi_std": round(float(np.std(s_vals)), 4),
                    "stoi_median": round(float(np.median(s_vals)), 4),
                    "delta_pesq_vs_noisy": round(p_mean - n_noisy, 4),
                    "delta_pesq_vs_zeroshot": round(p_mean - n_zs, 4),
                    "delta_stoi_vs_noisy": round(s_mean - n_noisy_s, 4),
                    "delta_stoi_vs_zeroshot": round(s_mean - n_zs_s, 4),
                }
                noise_table.append(row)
                print(f"  [{noise:7s}] {c:18s} | PESQ: {p_mean:.4f} | STOI: {s_mean:.4f} | dPESQ(zs): {row['delta_pesq_vs_zeroshot']:+.4f}")

        noise_csv_path = self.output_dir / "test_breakdown_by_noise.csv"
        with open(noise_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(noise_table[0].keys()))
            writer.writeheader()
            writer.writerows(noise_table)
        print(f"Saved noise breakdown to {noise_csv_path}.")

        # 5. Breakdown by SNR Level (2.5, 7.5, 12.5, 17.5 dB)
        print("\n--- BREAKDOWN BY SNR LEVEL ---")
        snr_table = []
        snr_levels = [2.5, 7.5, 12.5, 17.5]
        for snr in snr_levels:
            s_noisy = np.mean([float(r["pesq_wb"]) for r in condition_rows["noisy"] if float(r["snr_db"]) == snr])
            s_zs = np.mean([float(r["pesq_wb"]) for r in condition_rows["dns48_zeroshot"] if float(r["snr_db"]) == snr])
            s_noisy_s = np.mean([float(r["stoi"]) for r in condition_rows["noisy"] if float(r["snr_db"]) == snr])
            s_zs_s = np.mean([float(r["stoi"]) for r in condition_rows["dns48_zeroshot"] if float(r["snr_db"]) == snr])

            for c in conditions:
                rows = [r for r in condition_rows[c] if float(r["snr_db"]) == snr]
                p_vals = [float(r["pesq_wb"]) for r in rows]
                s_vals = [float(r["stoi"]) for r in rows]
                p_mean = float(np.mean(p_vals))
                s_mean = float(np.mean(s_vals))
                row = {
                    "snr_db": snr,
                    "condition": c,
                    "num_utterances": len(rows),
                    "pesq_mean": round(p_mean, 4),
                    "pesq_std": round(float(np.std(p_vals)), 4),
                    "pesq_median": round(float(np.median(p_vals)), 4),
                    "stoi_mean": round(s_mean, 4),
                    "stoi_std": round(float(np.std(s_vals)), 4),
                    "stoi_median": round(float(np.median(s_vals)), 4),
                    "delta_pesq_vs_noisy": round(p_mean - s_noisy, 4),
                    "delta_pesq_vs_zeroshot": round(p_mean - s_zs, 4),
                    "delta_stoi_vs_noisy": round(s_mean - s_noisy_s, 4),
                    "delta_stoi_vs_zeroshot": round(s_mean - s_zs_s, 4),
                }
                snr_table.append(row)
                print(f"  [{snr:4.1f} dB] {c:18s} | PESQ: {p_mean:.4f} | STOI: {s_mean:.4f} | dPESQ(zs): {row['delta_pesq_vs_zeroshot']:+.4f}")

        snr_csv_path = self.output_dir / "test_breakdown_by_snr.csv"
        with open(snr_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(snr_table[0].keys()))
            writer.writeheader()
            writer.writerows(snr_table)
        print(f"Saved SNR breakdown to {snr_csv_path}.")

        # 6. Utterance-Level Diagnostic Analysis
        print("\n--- UTTERANCE-LEVEL DIAGNOSTIC ANALYSIS ---")
        deltas_zs = []
        deltas_noisy = []
        cases_ft_gt_zs = 0
        cases_ft_le_zs = 0
        cases_ft_gt_noisy = 0
        cases_pesq_up_stoi_down_zs = 0
        cases_pesq_up_stoi_down_noisy = 0

        for uid, cond_map in by_utt.items():
            p_noisy = float(cond_map["noisy"]["pesq_wb"])
            s_noisy = float(cond_map["noisy"]["stoi"])
            p_zs = float(cond_map["dns48_zeroshot"]["pesq_wb"])
            s_zs = float(cond_map["dns48_zeroshot"]["stoi"])
            p_ft = float(cond_map["dns48_finetuned"]["pesq_wb"])
            s_ft = float(cond_map["dns48_finetuned"]["stoi"])

            d_p_zs = p_ft - p_zs
            d_p_noisy = p_ft - p_noisy

            deltas_zs.append({
                "utterance_id": uid,
                "speaker_id": cond_map["dns48_finetuned"]["speaker_id"],
                "noise_type": cond_map["dns48_finetuned"]["noise_type"],
                "snr_db": float(cond_map["dns48_finetuned"]["snr_db"]),
                "delta_pesq": round(d_p_zs, 4),
                "pesq_ft": round(p_ft, 4),
                "pesq_zs": round(p_zs, 4),
                "stoi_ft": round(s_ft, 4),
                "stoi_zs": round(s_zs, 4),
            })
            deltas_noisy.append({
                "utterance_id": uid,
                "speaker_id": cond_map["dns48_finetuned"]["speaker_id"],
                "noise_type": cond_map["dns48_finetuned"]["noise_type"],
                "snr_db": float(cond_map["dns48_finetuned"]["snr_db"]),
                "delta_pesq": round(d_p_noisy, 4),
                "pesq_ft": round(p_ft, 4),
                "pesq_noisy": round(p_noisy, 4),
                "stoi_ft": round(s_ft, 4),
                "stoi_noisy": round(s_noisy, 4),
            })

            if p_ft > p_zs:
                cases_ft_gt_zs += 1
            else:
                cases_ft_le_zs += 1

            if p_ft > p_noisy:
                cases_ft_gt_noisy += 1

            if p_ft > p_zs and s_ft < s_zs:
                cases_pesq_up_stoi_down_zs += 1

            if p_ft > p_noisy and s_ft < s_noisy:
                cases_pesq_up_stoi_down_noisy += 1

        deltas_zs.sort(key=lambda x: x["delta_pesq"], reverse=True)
        top5_zs = deltas_zs[:5]
        bottom5_zs = deltas_zs[-5:]

        deltas_noisy.sort(key=lambda x: x["delta_pesq"], reverse=True)
        top5_noisy = deltas_noisy[:5]
        bottom5_noisy = deltas_noisy[-5:]

        print(f"  Utterances with PESQ improvement over Zero-Shot: {cases_ft_gt_zs}/824 ({cases_ft_gt_zs/824*100:.1f}%)")
        print(f"  Utterances without PESQ improvement over Zero-Shot: {cases_ft_le_zs}/824 ({cases_ft_le_zs/824*100:.1f}%)")
        print(f"  Utterances with PESQ improvement over Noisy: {cases_ft_gt_noisy}/824 ({cases_ft_gt_noisy/824*100:.1f}%)")
        print(f"  PESQ improved but STOI reduced vs Zero-Shot: {cases_pesq_up_stoi_down_zs}/824 ({cases_pesq_up_stoi_down_zs/824*100:.1f}%)")
        print(f"  PESQ improved but STOI reduced vs Noisy: {cases_pesq_up_stoi_down_noisy}/824 ({cases_pesq_up_stoi_down_noisy/824*100:.1f}%)")

        # Smoke test utterance note
        smoke_uid = "p232_001"
        smoke_info = by_utt.get(smoke_uid, {})
        print(f"\n  [NOTE: Smoke-Test Utterance {smoke_uid}]")
        print(f"  Metadata: speaker={smoke_info.get('noisy', {}).get('speaker_id')}, "
              f"noise={smoke_info.get('noisy', {}).get('noise_type')}, "
              f"snr={smoke_info.get('noisy', {}).get('snr_db')} dB")
        print(f"  Scores: Noisy PESQ={smoke_info.get('noisy', {}).get('pesq_wb')}, "
              f"Spectral Gating PESQ={smoke_info.get('spectral_gating', {}).get('pesq_wb')}, "
              f"Zero-Shot PESQ={smoke_info.get('dns48_zeroshot', {}).get('pesq_wb')}, "
              f"Fine-Tuned PESQ={smoke_info.get('dns48_finetuned', {}).get('pesq_wb')}")

        # 7. Export 4 Representative Audio Examples
        print("\n--- EXPORTING REPRESENTATIVE AUDIO EXAMPLES ---")
        rep_specs = [
            {"snr_db": 2.5, "speaker_id": "p232", "noise_type": "psquare"},
            {"snr_db": 7.5, "speaker_id": "p257", "noise_type": "cafe"},
            {"snr_db": 12.5, "speaker_id": "p232", "noise_type": "living"},
            {"snr_db": 17.5, "speaker_id": "p257", "noise_type": "office"},
        ]

        # Load dataset items for audio extraction
        dataset = SpeechEnhancementDataset(
            manifest_path=self.test_manifest_path,
            local_data_dir=self.config.get("paths", {}).get("local_data_16k_dir", "/content/data_16k"),
            split="test",
            sample_rate=16000,
        )
        uid_to_idx = {entry["utterance_id"]: idx for idx, entry in enumerate(dataset.manifest)}

        # Initialize baseline models for export
        base_cfg = self.config.get("baseline", {})
        sg_params = {
            "sr": base_cfg.get("sr", 16000),
            "stationary": base_cfg.get("stationary", True),
            "n_fft": base_cfg.get("n_fft", 512),
            "win_length": base_cfg.get("win_length", 512),
            "hop_length": base_cfg.get("hop_length", 128),
            "n_std_thresh_stationary": base_cfg.get("n_std_thresh_stationary", 1.5),
            "prop_decrease": base_cfg.get("prop_decrease", 1.0),
        }
        sg_baseline = SpectralGatingBaseline(**sg_params)
        zs_model = load_pretrained_dns48(device=str(self.device))
        zs_model.eval()

        audio_examples_dir = self.output_dir / "audio_examples"
        audio_examples_dir.mkdir(parents=True, exist_ok=True)
        drive_audio_dir = Path("/content/drive/MyDrive/Speech-Enhancement/results/audio_examples")
        drive_audio_dir.mkdir(parents=True, exist_ok=True)

        selected_examples_metadata = []

        for spec in rep_specs:
            target_uid = None
            for item in dataset.manifest:
                if (float(item["snr_db"]) == spec["snr_db"] and
                    item["speaker_id"] == spec["speaker_id"] and
                    item["noise_type"] == spec["noise_type"]):
                    target_uid = item["utterance_id"]
                    break

            if target_uid is None:
                print(f"  [WARN] Could not find match for spec {spec}")
                continue

            item_data = dataset[uid_to_idx[target_uid]]
            clean_np = item_data["clean"].squeeze(0).numpy()
            noisy_np = item_data["noisy"].squeeze(0).numpy()

            # 1. Spectral Gating
            sg_np = sg_baseline.enhance(noisy_np)

            # 2. Demucs Zero-Shot
            with torch.no_grad():
                noisy_tensor = item_data["noisy"].unsqueeze(0).to(self.device)
                zs_np = zs_model(noisy_tensor).squeeze(0).squeeze(0).cpu().numpy()

            # 3. Demucs Fine-Tuned (from cache)
            ft_np = enhanced_cache[target_uid]

            # Save to local project and Drive
            example_subdirs = [audio_examples_dir / target_uid, drive_audio_dir / target_uid]
            for sdir in example_subdirs:
                sdir.mkdir(parents=True, exist_ok=True)
                save_wav_16k_pcm16(sdir / "clean.wav", clean_np)
                save_wav_16k_pcm16(sdir / "noisy.wav", noisy_np)
                save_wav_16k_pcm16(sdir / "spectral_gating.wav", sg_np)
                save_wav_16k_pcm16(sdir / "dns48_zeroshot.wav", zs_np)
                save_wav_16k_pcm16(sdir / "dns48_finetuned.wav", ft_np)

            m = by_utt[target_uid]
            example_meta = {
                "utterance_id": target_uid,
                "speaker_id": spec["speaker_id"],
                "noise_type": spec["noise_type"],
                "snr_db": spec["snr_db"],
                "duration_sec": item_data["duration_sec"],
                "clean_path": f"audio_examples/{target_uid}/clean.wav",
                "noisy_path": f"audio_examples/{target_uid}/noisy.wav",
                "spectral_gating_path": f"audio_examples/{target_uid}/spectral_gating.wav",
                "dns48_zeroshot_path": f"audio_examples/{target_uid}/dns48_zeroshot.wav",
                "dns48_finetuned_path": f"audio_examples/{target_uid}/dns48_finetuned.wav",
                "metrics": {
                    "noisy": {"pesq": m["noisy"]["pesq_wb"], "stoi": m["noisy"]["stoi"]},
                    "spectral_gating": {"pesq": m["spectral_gating"]["pesq_wb"], "stoi": m["spectral_gating"]["stoi"]},
                    "dns48_zeroshot": {"pesq": m["dns48_zeroshot"]["pesq_wb"], "stoi": m["dns48_zeroshot"]["stoi"]},
                    "dns48_finetuned": {"pesq": m["dns48_finetuned"]["pesq_wb"], "stoi": m["dns48_finetuned"]["stoi"]},
                }
            }
            selected_examples_metadata.append(example_meta)
            print(f"  Exported 5-audio set for {target_uid} [{spec['speaker_id']}, {spec['noise_type']}, {spec['snr_db']} dB]: "
                  f"Noisy={m['noisy']['pesq_wb']} -> ZS={m['dns48_zeroshot']['pesq_wb']} -> FT={m['dns48_finetuned']['pesq_wb']}")

        # 8. Publication Plot: final_test_comparison.png
        print("\n--- GENERATING PUBLICATION FIGURE ---")
        self._plot_publication_figure(summary_table, snr_table, noise_table, deltas_zs, deltas_noisy)

        # 9. Save final_test_comparison.json
        comprehensive_summary = {
            "evaluation_phase": "Phase 4 - Official Test-Set Evaluation",
            "checkpoint_used": {
                "path": str(self.checkpoint_path),
                "selection_criterion": "Best Validation PESQ (Epoch 14)",
                "val_pesq": 2.5691,
                "val_stoi": 0.8835,
            },
            "test_dataset": {
                "total_utterances": 824,
                "speakers": {"p232": 393, "p257": 431},
                "unseen_noises": noise_types,
                "snr_levels_db": snr_levels,
            },
            "overall_summary": summary_table,
            "speaker_breakdown": speaker_table,
            "noise_breakdown": noise_table,
            "snr_breakdown": snr_table,
            "diagnostic_analysis": {
                "counts": {
                    "pesq_improved_over_zeroshot_count": cases_ft_gt_zs,
                    "pesq_improved_over_zeroshot_pct": round(cases_ft_gt_zs / 824 * 100, 2),
                    "pesq_not_improved_over_zeroshot_count": cases_ft_le_zs,
                    "pesq_not_improved_over_zeroshot_pct": round(cases_ft_le_zs / 824 * 100, 2),
                    "pesq_improved_over_noisy_count": cases_ft_gt_noisy,
                    "pesq_improved_over_noisy_pct": round(cases_ft_gt_noisy / 824 * 100, 2),
                    "pesq_improved_but_stoi_reduced_vs_zeroshot_count": cases_pesq_up_stoi_down_zs,
                    "pesq_improved_but_stoi_reduced_vs_zeroshot_pct": round(cases_pesq_up_stoi_down_zs / 824 * 100, 2),
                    "pesq_improved_but_stoi_reduced_vs_noisy_count": cases_pesq_up_stoi_down_noisy,
                    "pesq_improved_but_stoi_reduced_vs_noisy_pct": round(cases_pesq_up_stoi_down_noisy / 824 * 100, 2),
                },
                "top5_improvements_vs_zeroshot": top5_zs,
                "bottom5_improvements_vs_zeroshot": bottom5_zs,
                "top5_improvements_vs_noisy": top5_noisy,
                "bottom5_improvements_vs_noisy": bottom5_noisy,
                "smoke_test_utterance_audit": {
                    "utterance_id": smoke_uid,
                    "note": "Utterance p232_001.wav was previously evaluated in Stage 2 single-pair smoke test for pipeline mechanics verification and is normally evaluated here.",
                    "metrics": {
                        "noisy": {"pesq": smoke_info.get("noisy", {}).get("pesq_wb"), "stoi": smoke_info.get("noisy", {}).get("stoi")},
                        "spectral_gating": {"pesq": smoke_info.get("spectral_gating", {}).get("pesq_wb"), "stoi": smoke_info.get("spectral_gating", {}).get("stoi")},
                        "dns48_zeroshot": {"pesq": smoke_info.get("dns48_zeroshot", {}).get("pesq_wb"), "stoi": smoke_info.get("dns48_zeroshot", {}).get("stoi")},
                        "dns48_finetuned": {"pesq": smoke_info.get("dns48_finetuned", {}).get("pesq_wb"), "stoi": smoke_info.get("dns48_finetuned", {}).get("stoi")},
                    }
                }
            },
            "selected_audio_examples": selected_examples_metadata,
        }

        final_json_path = self.output_dir / "final_test_comparison.json"
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(comprehensive_summary, f, indent=2)
        print(f"Saved comprehensive summary JSON to {final_json_path}.")

        # Also copy final_test_comparison.json, csv, and plot to Google Drive results
        drive_results_dir = Path("/content/drive/MyDrive/Speech-Enhancement/results")
        if drive_results_dir.exists():
            for fname in ["final_test_comparison.csv", "final_test_comparison.json", "final_test_comparison.png",
                          "test_finetuned_per_utterance.csv", "test_finetuned_summary.json",
                          "test_breakdown_by_speaker.csv", "test_breakdown_by_noise.csv", "test_breakdown_by_snr.csv"]:
                src_f = self.output_dir / fname
                if src_f.exists():
                    shutil.copy2(src_f, drive_results_dir / fname)
            print(f"Copied all Phase 4 result artifacts to persistent Drive: {drive_results_dir}")

        return comprehensive_summary

    def _plot_publication_figure(
        self,
        summary_table: List[Dict[str, Any]],
        snr_table: List[Dict[str, Any]],
        noise_table: List[Dict[str, Any]],
        deltas_zs: List[Dict[str, Any]],
        deltas_noisy: List[Dict[str, Any]],
    ) -> None:
        """Generates a high-resolution 4-panel publication plot comparing the 4 methods."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
        plt.subplots_adjust(hspace=0.32, wspace=0.25)

        colors = {
            "noisy": "#7f7f7f",            # Neutral gray
            "spectral_gating": "#d62728",  # Crimson
            "dns48_zeroshot": "#1f77b4",   # Steel blue
            "dns48_finetuned": "#2ca02c",  # Forest green
        }
        labels = {
            "noisy": "Noisy (Unprocessed)",
            "spectral_gating": "Spectral Gating",
            "dns48_zeroshot": "Demucs dns48 (Zero-Shot)",
            "dns48_finetuned": "Demucs dns48 (Fine-Tuned)",
        }
        cond_order = ["noisy", "spectral_gating", "dns48_zeroshot", "dns48_finetuned"]

        # ----------------------------------------------------
        # Panel 1: Overall Wideband PESQ & STOI (Bar Chart)
        # ----------------------------------------------------
        ax1 = axes[0, 0]
        x = np.arange(len(cond_order))
        width = 0.36

        pesq_means = [next(r["pesq_mean"] for r in summary_table if r["condition"] == c) for c in cond_order]
        pesq_stds = [next(r["pesq_std"] for r in summary_table if r["condition"] == c) for c in cond_order]
        stoi_means = [next(r["stoi_mean"] for r in summary_table if r["condition"] == c) for c in cond_order]
        stoi_stds = [next(r["stoi_std"] for r in summary_table if r["condition"] == c) for c in cond_order]

        ax1_b = ax1.twinx()

        bars_p = ax1.bar(x - width/2, pesq_means, width, yerr=pesq_stds, capsize=4,
                         color=[colors[c] for c in cond_order], alpha=0.88, edgecolor="black", label="Wideband PESQ")
        bars_s = ax1_b.bar(x + width/2, stoi_means, width, yerr=stoi_stds, capsize=4,
                           color=[colors[c] for c in cond_order], alpha=0.45, hatch="//", edgecolor="black", label="STOI")

        ax1.set_ylabel("Wideband PESQ (1.0 – 4.5)", fontsize=12, fontweight="bold", color="#1f77b4")
        ax1_b.set_ylabel("STOI (0.0 – 1.0)", fontsize=12, fontweight="bold", color="#2ca02c")
        ax1.set_ylim(0.0, 4.0)
        ax1_b.set_ylim(0.6, 1.02)
        ax1.set_xticks(x)
        ax1.set_xticklabels([labels[c] for c in cond_order], rotation=15, ha="right", fontsize=10, fontweight="bold")
        ax1.set_title("A) Overall Test-Set Performance (824 Utterances)", fontsize=13, fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        for b, v in zip(bars_p, pesq_means):
            ax1.text(b.get_x() + b.get_width()/2, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        for b, v in zip(bars_s, stoi_means):
            ax1_b.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # ----------------------------------------------------
        # Panel 2: Mean PESQ across Test SNR Levels (Line/Point Plot)
        # ----------------------------------------------------
        ax2 = axes[0, 1]
        snrs = [2.5, 7.5, 12.5, 17.5]
        for c in cond_order:
            means = [next(r["pesq_mean"] for r in snr_table if r["condition"] == c and float(r["snr_db"]) == s) for s in snrs]
            stds = [next(r["pesq_std"] for r in snr_table if r["condition"] == c and float(r["snr_db"]) == s) for s in snrs]
            ax2.errorbar(snrs, means, yerr=stds, marker="o", linewidth=2.4, markersize=7, capsize=4,
                         color=colors[c], label=labels[c])

        ax2.set_xlabel("Test Input SNR (dB)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Wideband PESQ", fontsize=12, fontweight="bold")
        ax2.set_title("B) Enhancement Performance across Input SNR Levels", fontsize=13, fontweight="bold")
        ax2.set_xticks(snrs)
        ax2.set_ylim(1.0, 3.8)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right", frameon=True, fontsize=10)

        # ----------------------------------------------------
        # Panel 3: Performance across 5 Unseen Noise Types
        # ----------------------------------------------------
        ax3 = axes[1, 0]
        noises = ["bus", "cafe", "living", "office", "psquare"]
        x_n = np.arange(len(noises))
        w_n = 0.18

        for idx, c in enumerate(cond_order):
            means = [next(r["pesq_mean"] for r in noise_table if r["condition"] == c and r["noise_type"] == n) for n in noises]
            ax3.bar(x_n + (idx - 1.5) * w_n, means, w_n, color=colors[c], label=labels[c], edgecolor="black", alpha=0.9)

        ax3.set_xlabel("Unseen DEMAND Noise Type", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Wideband PESQ", fontsize=12, fontweight="bold")
        ax3.set_title("C) Generalization across 5 Unseen Noise Types", fontsize=13, fontweight="bold")
        ax3.set_xticks(x_n)
        ax3.set_xticklabels([f"{n.upper()}" for n in noises], fontsize=11, fontweight="bold")
        ax3.set_ylim(1.0, 3.8)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)
        ax3.legend(loc="upper left", frameon=True, fontsize=9)

        # ----------------------------------------------------
        # Panel 4: Per-Utterance Delta PESQ Distribution
        # ----------------------------------------------------
        ax4 = axes[1, 1]
        dp_zs = [d["delta_pesq"] for d in deltas_zs]
        dp_noisy = [d["delta_pesq"] for d in deltas_noisy]

        bins = np.linspace(-0.8, 1.8, 45)
        ax4.hist(dp_noisy, bins=bins, color=colors["noisy"], alpha=0.5, edgecolor="black", label=r"$\Delta$PESQ (Fine-Tuned vs. Noisy)")
        ax4.hist(dp_zs, bins=bins, color=colors["dns48_finetuned"], alpha=0.65, edgecolor="black", label=r"$\Delta$PESQ (Fine-Tuned vs. Zero-Shot)")

        ax4.axvline(0.0, color="red", linestyle="--", linewidth=1.8, label="Zero-Delta Reference")
        median_zs = np.median(dp_zs)
        median_noisy = np.median(dp_noisy)
        ax4.axvline(median_zs, color="green", linestyle="-.", linewidth=2.0, label=f"Median vs. ZS ({median_zs:+.2f})")
        ax4.axvline(median_noisy, color="blue", linestyle=":", linewidth=2.0, label=f"Median vs. Noisy ({median_noisy:+.2f})")

        ax4.set_xlabel(r"Per-Utterance $\Delta$PESQ", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Utterance Count (out of 824)", fontsize=12, fontweight="bold")
        ax4.set_title(r"D) Distribution of Per-Utterance $\Delta$PESQ Improvements", fontsize=13, fontweight="bold")
        ax4.grid(axis="y", linestyle="--", alpha=0.5)
        ax4.legend(loc="upper right", frameon=True, fontsize=10)

        plt.suptitle("Speech Enhancement Official Test-Set Evaluation (VoiceBank-DEMAND, 824 Utterances)",
                     fontsize=16, fontweight="bold", y=0.99)

        plot_path = self.output_dir / "final_test_comparison.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved publication comparison figure to {plot_path}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned Demucs dns48 on Official Test Set.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/Speech-Enhancement/checkpoints/best_val_pesq.th",
        help="Path to best validation PESQ checkpoint.",
    )
    parser.add_argument("--test-manifest", type=str, default="manifests/test_manifest.json", help="Path to test manifest JSON.")
    parser.add_argument("--baseline-csv", type=str, default="results/test_baseline_per_utterance.csv", help="Phase 3A baseline CSV.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output reports and plots.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device ('cuda' or 'cpu').")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight checks only and exit.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Local fallback for checkpoint if Drive path is not directly mounted
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        local_fallback = Path("checkpoints/best_val_pesq.th")
        if local_fallback.exists():
            print(f"[FALLBACK] Using local checkpoint path: {local_fallback}")
            ckpt_path = local_fallback

    evaluator = OfficialTestEvaluator(
        config=config,
        checkpoint_path=ckpt_path,
        test_manifest_path=args.test_manifest,
        baseline_csv_path=args.baseline_csv,
        output_dir=args.output_dir,
        device=args.device,
    )

    # 1. Run controlled preflight check
    evaluator.run_preflight_check()

    if args.preflight_only:
        print("[INFO] Preflight checks passed. Exiting (--preflight-only specified).")
        return

    # 2. Run full 824-utterance evaluation
    finetuned_rows, enhanced_cache = evaluator.evaluate_test_set()

    # 3. Compile unified 4-way comparative analysis, breakdowns, plot, and audio exports
    evaluator.compile_comparative_analysis(finetuned_rows, enhanced_cache)

    print("\n" + "=" * 70)
    print("PHASE 4 OFFICIAL TEST-SET EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
