"""Demucs dns48 Fine-Tuning Trainer with Multi-Resolution STFT Loss and Validation."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import SpeechEnhancementDataset, get_dataloader, ensure_local_data_16k
from src.evaluation.evaluator import compute_utterance_metrics
from src.models.demucs_loader import load_pretrained_dns48, save_checkpoint_atomic, load_checkpoint
from src.training.stft_loss import MultiResolutionSTFTLoss
from src.utils.seed import set_global_seed


def _val_metrics_worker(args: Tuple[str, np.ndarray, np.ndarray, int]) -> Dict[str, Any]:
    """Worker function for parallel PESQ and STOI computation during validation."""
    utt_id, clean_np, est_np, sr = args
    pesq_val, stoi_val, pesq_err, stoi_err = compute_utterance_metrics(clean_np, est_np, sr=sr)
    return {
        "utterance_id": utt_id,
        "pesq": pesq_val,
        "stoi": stoi_val,
        "pesq_err": pesq_err,
        "stoi_err": stoi_err,
    }


class DemucsTrainer:
    """Trainer for fine-tuning Demucs dns48 on Edinburgh VoiceBank-DEMAND."""

    def __init__(
        self,
        config: Dict[str, Any],
        train_manifest: Union[str, Path] = "manifests/train_manifest.json",
        val_manifest: Union[str, Path] = "manifests/val_manifest.json",
        device: Optional[str] = None,
        checkpoints_dir: Optional[Union[str, Path]] = None,
        results_dir: Optional[Union[str, Path]] = None,
        local_data_dir: Optional[Union[str, Path]] = None,
        tar_path: Optional[Union[str, Path]] = None,
        max_workers: Optional[int] = None,
    ):
        self.config = config
        self.seed = self.config.get("project", {}).get("seed", 42)
        set_global_seed(self.seed)

        paths_cfg = self.config.get("paths", {})
        self.checkpoints_dir = Path(checkpoints_dir or paths_cfg.get("checkpoints_dir", "/content/drive/MyDrive/Speech-Enhancement/checkpoints"))
        self.results_dir = Path(results_dir or paths_cfg.get("results_dir", "results"))
        self.local_data_dir = Path(local_data_dir or paths_cfg.get("local_data_16k_dir", "/content/data_16k"))
        self.tar_path = Path(tar_path or paths_cfg.get("tar_16k_path", "/content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar"))

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.max_workers = max_workers or min(os.cpu_count() or 4, 8)

        # Auto-restore check for Colab NVMe scratch storage
        if str(self.local_data_dir).startswith("/content"):
            ensure_local_data_16k(self.local_data_dir, tar_path=self.tar_path)

        # Hyperparameters
        train_cfg = self.config.get("training", {})
        self.lr = float(train_cfg.get("lr", 5e-5))
        self.betas = tuple(train_cfg.get("betas", [0.9, 0.999]))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.0))
        self.batch_size = int(train_cfg.get("batch_size", 16))
        self.num_epochs = int(train_cfg.get("epochs", 15))
        self.grad_clip_max_norm = float(train_cfg.get("grad_clip_max_norm", 5.0))
        self.num_workers = int(train_cfg.get("num_workers", 2))

        # Loss weights
        loss_cfg = self.config.get("loss", {})
        self.l1_weight = float(loss_cfg.get("l1_weight", 1.0))
        self.stft_sc_factor = float(loss_cfg.get("stft_sc_factor", 0.1))
        self.stft_mag_factor = float(loss_cfg.get("stft_mag_factor", 0.1))
        fft_sizes = loss_cfg.get("fft_sizes", [1024, 2048, 512])
        hop_sizes = loss_cfg.get("hop_sizes", [120, 240, 50])
        win_lengths = loss_cfg.get("win_lengths", [600, 1200, 240])

        # Loss modules
        self.mr_stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
        ).to(self.device)

        # Model setup
        print(f"Loading pretrained Demucs dns48 on {self.device}...")
        self.model = load_pretrained_dns48(device=self.device)

        # Optimizer setup
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        # Dataset & Dataloaders setup
        self.train_manifest = Path(train_manifest)
        self.val_manifest = Path(val_manifest)
        self._setup_dataloaders()

        # Training history & metric tracking
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.best_val_loss_epoch = 0
        self.best_val_pesq = -1.0
        self.best_val_pesq_epoch = 0
        self.best_val_stoi = -1.0
        self.history: List[Dict[str, Any]] = []

    def _setup_dataloaders(self) -> None:
        """Initializes train and validation datasets and asserts strict speaker isolation."""
        ds_cfg = self.config.get("dataset", {})
        segment_sec = float(ds_cfg.get("segment_seconds", 4.0))
        stride_sec = float(ds_cfg.get("stride_seconds", 2.0))
        sample_rate = int(ds_cfg.get("sample_rate", 16000))
        test_speakers = set(ds_cfg.get("test_speakers", ["p232", "p257"]))

        # Build train dataset (sliding-window chunked mode)
        self.train_dataset = SpeechEnhancementDataset(
            manifest_path=self.train_manifest,
            local_data_dir=self.local_data_dir,
            split="train",
            sample_rate=sample_rate,
            segment_seconds=segment_sec,
            stride_seconds=stride_sec,
            pad=True,
            auto_restore=False,
        )

        # Build validation dataset (full uncropped mode)
        self.val_dataset = SpeechEnhancementDataset(
            manifest_path=self.val_manifest,
            local_data_dir=self.local_data_dir,
            split="val",
            sample_rate=sample_rate,
            pad=False,
            auto_restore=False,
        )

        # Programmatic Assertions: Zero Speaker Leakage
        train_spks = set(item["speaker_id"] for item in self.train_dataset.manifest)
        val_spks = set(item["speaker_id"] for item in self.val_dataset.manifest)

        leak_train_test = train_spks & test_speakers
        leak_val_test = val_spks & test_speakers
        leak_train_val = train_spks & val_spks

        if leak_train_test:
            raise ValueError(f"FATAL: Test speakers found in training manifest: {leak_train_test}")
        if leak_val_test:
            raise ValueError(f"FATAL: Test speakers found in validation manifest: {leak_val_test}")
        if leak_train_val:
            raise ValueError(f"FATAL: Validation speakers found in training manifest: {leak_train_val}")

        print(f"Dataset verification passed: {len(train_spks)} train speakers, {len(val_spks)} val speakers, 0 test overlap.")
        print(f"Train utterances: {len(self.train_dataset.manifest)} -> {len(self.train_dataset)} sliding-window chunks (4.0s/stride 2.0s).")
        print(f"Val utterances: {len(self.val_dataset)} full-length utterances (held-out speakers: {sorted(val_spks)}).")

        self.train_loader = get_dataloader(
            manifest_path=self.train_manifest,
            local_data_dir=self.local_data_dir,
            split="train",
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            segment_seconds=segment_sec,
            stride_seconds=stride_sec,
            pad=True,
            auto_restore=False,
            seed=self.seed,
        )

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs one full training epoch (767 batches @ batch_size=16)."""
        self.model.train()

        # Seed dataloader shuffling deterministically per epoch
        if hasattr(self.train_loader, "generator") and self.train_loader.generator is not None:
            self.train_loader.generator.manual_seed(self.seed + epoch)

        total_batches = len(self.train_loader)
        epoch_l1 = 0.0
        epoch_sc = 0.0
        epoch_mag = 0.0
        epoch_total = 0.0
        epoch_start = time.time()
        last_log_time = epoch_start

        for batch_idx, batch in enumerate(self.train_loader):
            clean = batch["clean"].to(self.device)  # [B, 1, 64000]
            noisy = batch["noisy"].to(self.device)  # [B, 1, 64000]

            self.optimizer.zero_grad(set_to_none=True)
            est = self.model(noisy)

            l1_loss = F.l1_loss(est, clean) * self.l1_weight
            raw_sc_loss, raw_mag_loss = self.mr_stft_loss(est, clean)

            # Combined loss: weights applied exactly once
            total_loss = l1_loss + self.stft_sc_factor * raw_sc_loss + self.stft_mag_factor * raw_mag_loss

            if not torch.isfinite(total_loss):
                raise ValueError(
                    f"Epoch {epoch} Batch {batch_idx}: non-finite loss detected (total={total_loss.item()}, "
                    f"l1={l1_loss.item()}, sc={raw_sc_loss.item()}, mag={raw_mag_loss.item()})"
                )

            total_loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
            if not torch.isfinite(grad_norm):
                raise ValueError(f"Epoch {epoch} Batch {batch_idx}: non-finite gradient norm {grad_norm.item()}")

            self.optimizer.step()

            # Accumulate
            b_l1 = l1_loss.item()
            b_sc = raw_sc_loss.item()
            b_mag = raw_mag_loss.item()
            b_tot = total_loss.item()

            epoch_l1 += b_l1
            epoch_sc += b_sc
            epoch_mag += b_mag
            epoch_total += b_tot

            curr_time = time.time()
            if (curr_time - last_log_time >= 15.0) or (batch_idx + 1 == total_batches):
                elapsed = curr_time - epoch_start
                rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"  Epoch {epoch:02d} [{batch_idx+1:03d}/{total_batches:03d}] - "
                    f"Loss: {b_tot:.4f} (L1: {b_l1:.4f}, SC: {b_sc:.3f}, Mag: {b_mag:.3f}) - "
                    f"GradNorm: {grad_norm.item():.2f} - {rate:.1f} batches/s"
                )
                last_log_time = curr_time

        return {
            "train_loss": epoch_total / total_batches,
            "train_l1_loss": epoch_l1 / total_batches,
            "train_sc_loss": epoch_sc / total_batches,
            "train_mag_loss": epoch_mag / total_batches,
            "epoch_duration_sec": round(time.time() - epoch_start, 2),
        }

    def validate(self, epoch: int) -> Dict[str, Any]:
        """Runs full validation pass on all 770 uncropped utterances and computes PESQ/STOI."""
        self.model.eval()
        val_start = time.time()

        val_l1_acc = 0.0
        val_sc_acc = 0.0
        val_mag_acc = 0.0
        val_total_acc = 0.0
        num_items = len(self.val_dataset)

        metric_tasks = []

        print(f"Running validation forward pass for Epoch {epoch} on {self.device} (770 utterances)...")
        with torch.no_grad():
            for i in range(num_items):
                item = self.val_dataset[i]
                clean = item["clean"].unsqueeze(0).to(self.device)  # [1, 1, T]
                noisy = item["noisy"].unsqueeze(0).to(self.device)  # [1, 1, T]

                est = self.model(noisy)

                l1 = F.l1_loss(est, clean) * self.l1_weight
                sc, mag = self.mr_stft_loss(est, clean)
                tot = l1 + self.stft_sc_factor * sc + self.stft_mag_factor * mag

                val_l1_acc += l1.item()
                val_sc_acc += sc.item()
                val_mag_acc += mag.item()
                val_total_acc += tot.item()

                clean_np = clean.squeeze(0).squeeze(0).cpu().numpy()
                est_np = est.squeeze(0).squeeze(0).cpu().numpy()
                metric_tasks.append((item["utterance_id"], clean_np, est_np, self.val_dataset.sample_rate))

        # Parallel PESQ / STOI computation using ProcessPoolExecutor
        pesq_scores = []
        stoi_scores = []
        pesq_errors = 0
        stoi_errors = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_val_metrics_worker, t) for t in metric_tasks]
            for fut in as_completed(futures):
                res = fut.result()
                if res["pesq"] is not None:
                    pesq_scores.append(res["pesq"])
                else:
                    pesq_errors += 1
                if res["stoi"] is not None:
                    stoi_scores.append(res["stoi"])
                else:
                    stoi_errors += 1

        val_duration = time.time() - val_start

        pesq_mean = float(np.mean(pesq_scores)) if pesq_scores else 0.0
        pesq_median = float(np.median(pesq_scores)) if pesq_scores else 0.0
        pesq_std = float(np.std(pesq_scores)) if pesq_scores else 0.0

        stoi_mean = float(np.mean(stoi_scores)) if stoi_scores else 0.0
        stoi_median = float(np.median(stoi_scores)) if stoi_scores else 0.0
        stoi_std = float(np.std(stoi_scores)) if stoi_scores else 0.0

        val_metrics = {
            "val_loss": val_total_acc / num_items,
            "val_l1_loss": val_l1_acc / num_items,
            "val_sc_loss": val_sc_acc / num_items,
            "val_mag_loss": val_mag_acc / num_items,
            "val_pesq_mean": round(pesq_mean, 4),
            "val_pesq_median": round(pesq_median, 4),
            "val_pesq_std": round(pesq_std, 4),
            "val_stoi_mean": round(stoi_mean, 4),
            "val_stoi_median": round(stoi_median, 4),
            "val_stoi_std": round(stoi_std, 4),
            "val_pesq_errors": pesq_errors,
            "val_stoi_errors": stoi_errors,
            "val_duration_sec": round(val_duration, 2),
        }

        print(
            f"  Val Epoch {epoch:02d}: Loss={val_metrics['val_loss']:.4f} | "
            f"PESQ={val_metrics['val_pesq_mean']:.4f} (med: {val_metrics['val_pesq_median']:.4f}) | "
            f"STOI={val_metrics['val_stoi_mean']:.4f} (med: {val_metrics['val_stoi_median']:.4f}) "
            f"[{val_duration:.1f}s]"
        )

        return val_metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        is_best_loss: bool = False,
        is_best_pesq: bool = False,
    ) -> None:
        """Atomically saves checkpoint to Google Drive and updates summary files."""
        extra_info = {
            "best_val_loss": self.best_val_loss,
            "best_val_loss_epoch": self.best_val_loss_epoch,
            "best_val_pesq": self.best_val_pesq,
            "best_val_pesq_epoch": self.best_val_pesq_epoch,
            "best_val_stoi": self.best_val_stoi,
            "latest_metrics": metrics,
        }

        # 1. Latest checkpoint (checkpoint.th)
        latest_path = self.checkpoints_dir / "checkpoint.th"
        save_checkpoint_atomic(
            checkpoint_path=latest_path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            history=self.history,
            extra=extra_info,
        )

        # 2. Best validation loss checkpoint
        if is_best_loss:
            best_loss_path = self.checkpoints_dir / "best_val_loss.th"
            save_checkpoint_atomic(
                checkpoint_path=best_loss_path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                history=self.history,
                extra=extra_info,
            )
            print(f"  Saved new best validation loss checkpoint: {best_loss_path.name} (val_loss={metrics['val_loss']:.4f})")

        # 3. Best validation PESQ checkpoint
        if is_best_pesq:
            best_pesq_path = self.checkpoints_dir / "best_val_pesq.th"
            save_checkpoint_atomic(
                checkpoint_path=best_pesq_path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                history=self.history,
                extra=extra_info,
            )
            print(f"  Saved new best validation PESQ checkpoint: {best_pesq_path.name} (val_pesq={metrics['val_pesq_mean']:.4f})")

    def resume_from_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None) -> bool:
        """Resumes model, optimizer, epoch, and history from checkpoint.th if available."""
        target_path = Path(checkpoint_path or (self.checkpoints_dir / "checkpoint.th"))
        if not target_path.exists():
            print(f"No checkpoint found at {target_path}. Starting training from scratch.")
            return False

        print(f"Resuming training from checkpoint: {target_path}...")
        payload = load_checkpoint(
            checkpoint_path=target_path,
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
        )

        self.start_epoch = payload.get("epoch", 0) + 1
        self.history = payload.get("history", [])

        extra = payload.get("extra", {})
        self.best_val_loss = extra.get("best_val_loss", float("inf"))
        self.best_val_loss_epoch = extra.get("best_val_loss_epoch", 0)
        self.best_val_pesq = extra.get("best_val_pesq", -1.0)
        self.best_val_pesq_epoch = extra.get("best_val_pesq_epoch", 0)
        self.best_val_stoi = extra.get("best_val_stoi", -1.0)

        print(
            f"Successfully resumed at Epoch {self.start_epoch}. Previous best val_loss: {self.best_val_loss:.4f} "
            f"(Ep {self.best_val_loss_epoch}), best val_pesq: {self.best_val_pesq:.4f} (Ep {self.best_val_pesq_epoch})."
        )
        return True

    def update_history_and_plots(self, epoch_entry: Dict[str, Any]) -> None:
        """Updates results/training_history.csv, results/training_summary.json, and results/training_curves.png."""
        self.history.append(epoch_entry)

        # 1. Update CSV
        csv_path = self.results_dir / "training_history.csv"
        fieldnames = list(epoch_entry.keys())
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(epoch_entry)

        # 2. Update summary JSON
        summary_path = self.results_dir / "training_summary.json"
        summary_data = {
            "total_epochs_trained": len(self.history),
            "current_epoch": epoch_entry["epoch"],
            "best_val_loss": self.best_val_loss,
            "best_val_loss_epoch": self.best_val_loss_epoch,
            "best_val_pesq": self.best_val_pesq,
            "best_val_pesq_epoch": self.best_val_pesq_epoch,
            "best_val_stoi": self.best_val_stoi,
            "history": self.history,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # 3. Generate Training Curves Plot
        self._plot_training_curves()

    def _plot_training_curves(self) -> None:
        """Plots training vs. validation loss, PESQ, and STOI curves with reference lines."""
        if not self.history:
            return

        epochs = [h["epoch"] for h in self.history]
        train_losses = [h.get("train_loss") for h in self.history]
        val_losses = [h.get("val_loss") for h in self.history]
        val_pesqs = [h.get("val_pesq_mean") for h in self.history]
        val_stois = [h.get("val_stoi_mean") for h in self.history]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Subplot 1: Losses
        ax1.plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=2)
        ax1.plot(epochs, val_losses, "r-s", label="Val Loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss ($L_1 + 0.1 \\times SC + 0.1 \\times Mag$)")
        ax1.set_title("Training & Validation Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # Subplot 2: PESQ
        ax2.plot(epochs, val_pesqs, "g-^", label="Val PESQ (wb)", linewidth=2)
        ax2.axhline(1.5714, color="gray", linestyle=":", label="Noisy Val Baseline (1.5714)")
        ax2.axhline(2.0691, color="orange", linestyle="--", label="Zero-Shot dns48 Baseline (2.0691)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("PESQ (Wideband)")
        ax2.set_title("Validation PESQ Progression")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        # Subplot 3: STOI
        ax3.plot(epochs, val_stois, "m-d", label="Val STOI", linewidth=2)
        ax3.axhline(0.8126, color="gray", linestyle=":", label="Noisy Val Baseline (0.8126)")
        ax3.axhline(0.8523, color="orange", linestyle="--", label="Zero-Shot dns48 Baseline (0.8523)")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("STOI")
        ax3.set_title("Validation STOI Progression")
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend()

        plt.tight_layout()
        plot_path = self.results_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

    def run_preflight(self) -> Dict[str, Any]:
        """Performs a controlled 10-batch preflight test and GPU profiling on Colab."""
        print("=" * 78)
        print("DEMUCS FINE-TUNING PREFLIGHT & GPU PROFILING")
        print("=" * 78)

        # 1. Device check
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"Active Device: {self.device} ({gpu_name})")

        # 2. Reset GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # 3. Mini training test (10 batches)
        self.model.train()
        print("\nRunning 10-batch mini training pass with batch_size=16...")
        batch_times = []
        test_loader = iter(self.train_loader)

        for b_idx in range(10):
            t0 = time.time()
            batch = next(test_loader)
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            est = self.model(noisy)

            l1 = F.l1_loss(est, clean) * self.l1_weight
            sc, mag = self.mr_stft_loss(est, clean)
            loss = l1 + self.stft_sc_factor * sc + self.stft_mag_factor * mag

            assert torch.isfinite(loss), f"Preflight batch {b_idx} non-finite loss"
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
            assert torch.isfinite(grad_norm), f"Preflight batch {b_idx} non-finite grad norm"

            self.optimizer.step()
            dt = time.time() - t0
            batch_times.append(dt)
            print(f"  Batch {b_idx+1:02d}/10: loss={loss.item():.4f}, grad_norm={grad_norm.item():.2f}, time={dt*1000:.1f}ms")

        avg_batch_ms = float(np.mean(batch_times)) * 1000
        batches_per_epoch = len(self.train_loader)  # 767 batches
        est_epoch_train_sec = (avg_batch_ms / 1000.0) * batches_per_epoch

        # 4. GPU Memory Profile
        peak_mem_mb = 0.0
        reserved_mem_mb = 0.0
        if torch.cuda.is_available():
            peak_mem_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            reserved_mem_mb = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2)
        print(f"\nGPU Peak Allocated Memory : {peak_mem_mb} MB")
        print(f"GPU Peak Reserved Memory  : {reserved_mem_mb} MB")

        # 5. Atomic checkpoint save and reload verification
        print("\nVerifying atomic checkpoint save and reload...")
        test_ckpt_path = self.results_dir / "preflight_checkpoint.tmp"
        save_checkpoint_atomic(
            checkpoint_path=test_ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=0,
            history=[],
            extra={"test": True},
        )
        assert test_ckpt_path.exists(), "Temporary checkpoint file was not created"
        load_checkpoint(test_ckpt_path, model=self.model, optimizer=self.optimizer, device=self.device)
        test_ckpt_path.unlink(missing_ok=True)
        print("  PASS: Atomic checkpoint save and reload succeeded.")

        # 6. Estimate validation duration
        # Based on Phase 3A, 770 val utterances forward pass + PESQ/STOI takes ~45 seconds
        est_val_sec = 45.0
        est_epoch_total_sec = est_epoch_train_sec + est_val_sec
        est_15_epochs_min = round((est_epoch_total_sec * self.num_epochs) / 60.0, 2)

        report = {
            "status": "PREFLIGHT_PASSED",
            "device": str(self.device),
            "gpu_name": gpu_name,
            "batch_size": self.batch_size,
            "batches_per_epoch": batches_per_epoch,
            "avg_batch_ms": round(avg_batch_ms, 2),
            "peak_gpu_allocated_mb": peak_mem_mb,
            "peak_gpu_reserved_mb": reserved_mem_mb,
            "est_epoch_train_sec": round(est_epoch_train_sec, 2),
            "est_epoch_val_sec": round(est_val_sec, 2),
            "est_epoch_total_sec": round(est_epoch_total_sec, 2),
            "est_15_epochs_total_minutes": est_15_epochs_min,
            "loss_weights": {
                "l1_weight": self.l1_weight,
                "stft_sc_factor": self.stft_sc_factor,
                "stft_mag_factor": self.stft_mag_factor,
            },
            "grad_clip_max_norm": self.grad_clip_max_norm,
            "dataset_counts": {
                "train_utterances": len(self.train_dataset.manifest),
                "train_windowed_chunks": len(self.train_dataset),
                "val_utterances": len(self.val_dataset),
            },
        }

        report_file = self.results_dir / "preflight_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved preflight report to {report_file}")
        print("=" * 78)
        print(f"PREFLIGHT SUMMARY: 1 Epoch ~ {est_epoch_total_sec:.1f}s | 15 Epochs ~ {est_15_epochs_min:.1f} minutes")
        print("=" * 78)
        return report

    def train(self) -> Dict[str, Any]:
        """Executes full fine-tuning training loop across all configured epochs."""
        print("=" * 78)
        print(f"STARTING DEMUCS DNS48 FINE-TUNING ({self.num_epochs} EPOCHS)")
        print("=" * 78)

        # Step 0: Pretrained Zero-Shot Validation Evaluation (Epoch 0)
        if self.start_epoch == 1 and not self.history:
            print("\n[EPOCH 00] Evaluating Pretrained Zero-Shot Demucs dns48 on Validation Set...")
            val_0 = self.validate(epoch=0)
            self.best_val_loss = val_0["val_loss"]
            self.best_val_loss_epoch = 0
            self.best_val_pesq = val_0["val_pesq_mean"]
            self.best_val_pesq_epoch = 0
            self.best_val_stoi = val_0["val_stoi_mean"]

            epoch_0_entry = {
                "epoch": 0,
                "train_loss": None,
                "train_l1_loss": None,
                "train_sc_loss": None,
                "train_mag_loss": None,
                "val_loss": val_0["val_loss"],
                "val_l1_loss": val_0["val_l1_loss"],
                "val_sc_loss": val_0["val_sc_loss"],
                "val_mag_loss": val_0["val_mag_loss"],
                "val_pesq_mean": val_0["val_pesq_mean"],
                "val_pesq_median": val_0["val_pesq_median"],
                "val_pesq_std": val_0["val_pesq_std"],
                "val_stoi_mean": val_0["val_stoi_mean"],
                "val_stoi_median": val_0["val_stoi_median"],
                "val_stoi_std": val_0["val_stoi_std"],
                "lr": self.lr,
                "epoch_duration_sec": val_0["val_duration_sec"],
                "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
            }
            self.update_history_and_plots(epoch_0_entry)
            self.save_checkpoint(epoch=0, metrics=val_0, is_best_loss=True, is_best_pesq=True)

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            print(f"\n{'='*30} EPOCH {epoch:02d}/{self.num_epochs:02d} {'='*30}")
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            # Determine best checkpoints
            is_best_loss = val_metrics["val_loss"] < self.best_val_loss
            if is_best_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_val_loss_epoch = epoch

            is_best_pesq = val_metrics["val_pesq_mean"] > self.best_val_pesq
            if is_best_pesq:
                self.best_val_pesq = val_metrics["val_pesq_mean"]
                self.best_val_pesq_epoch = epoch
                self.best_val_stoi = val_metrics["val_stoi_mean"]

            gpu_peak = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

            epoch_entry = {
                "epoch": epoch,
                "train_loss": round(train_metrics["train_loss"], 4),
                "train_l1_loss": round(train_metrics["train_l1_loss"], 4),
                "train_sc_loss": round(train_metrics["train_sc_loss"], 4),
                "train_mag_loss": round(train_metrics["train_mag_loss"], 4),
                "val_loss": round(val_metrics["val_loss"], 4),
                "val_l1_loss": round(val_metrics["val_l1_loss"], 4),
                "val_sc_loss": round(val_metrics["val_sc_loss"], 4),
                "val_mag_loss": round(val_metrics["val_mag_loss"], 4),
                "val_pesq_mean": val_metrics["val_pesq_mean"],
                "val_pesq_median": val_metrics["val_pesq_median"],
                "val_pesq_std": val_metrics["val_pesq_std"],
                "val_stoi_mean": val_metrics["val_stoi_mean"],
                "val_stoi_median": val_metrics["val_stoi_median"],
                "val_stoi_std": val_metrics["val_stoi_std"],
                "lr": self.lr,
                "epoch_duration_sec": train_metrics["epoch_duration_sec"] + val_metrics["val_duration_sec"],
                "gpu_peak_memory_mb": gpu_peak,
            }

            self.update_history_and_plots(epoch_entry)
            self.save_checkpoint(epoch=epoch, metrics=val_metrics, is_best_loss=is_best_loss, is_best_pesq=is_best_pesq)

        print("\n" + "=" * 78)
        print("ALL 15 FINE-TUNING EPOCHS COMPLETED SUCCESSFULLY!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f} (Epoch {self.best_val_loss_epoch})")
        print(f"Best Validation PESQ: {self.best_val_pesq:.4f} (Epoch {self.best_val_pesq_epoch})")
        print("=" * 78)

        return {
            "status": "TRAINING_COMPLETED",
            "epochs": self.num_epochs,
            "best_val_loss": self.best_val_loss,
            "best_val_loss_epoch": self.best_val_loss_epoch,
            "best_val_pesq": self.best_val_pesq,
            "best_val_pesq_epoch": self.best_val_pesq_epoch,
            "best_val_stoi": self.best_val_stoi,
            "history": self.history,
        }
