> **Document ID:** `11` | **Phase:** Phase 3B — Demucs dns48 Fine-Tuning | **Role:** Implementation Engineer Report  
> **Source Word Document:** `Phase_3B_Demucs_FineTuning_Final_Report.docx` | **Repository Path:** `Speech-Ench-Answers/11_Phase_3B_Demucs_FineTuning_Final_Report.md`

---
# Phase 3B Final Report: Demucs dns48 Fine-Tuning

**Executed By:** Implementation Engineer Subagent (impl-engineer, Model: Pro, Workspace: inherit, Write enabled, Colab NVIDIA L4 GPU)

## 1. Files Created or Modified

- **Created Training & Runner Modules:** `src/training/trainer.py (DemucsTrainer: deterministic seeding, automatic /content/data_16k restoration from edinburgh_16k.tar, strict speaker-isolation assertions, controlled preflight & GPU profiler, L1 + 0.1 * SC + 0.1 * Mag loss loop, gradient clipping max_norm=5.0, parallel 770-utterance validation PESQ/STOI evaluation, and atomic checkpointing) and scripts/06_finetune_demucs.py (CLI runner supporting --preflight-only, full 15-epoch training, and --resume).`
- **Modified Package Export & Documentation:** `src/training/__init__.py and README.md.`
- **Generated Phase 3B Artifacts in results/:** `results/preflight_report.json, results/training_history.csv (Epoch 00 zero-shot + Epochs 01–15), results/training_summary.json, and results/training_curves.png.`
- **Unmodified Core Files (Verified Intact):** src/models/demucs_loader.py, src/training/stft_loss.py, configs/config.yaml, and all manifests/*.json / .csv files were strictly preserved without modification.

## 2. Exact Training Configuration

Verified directly against configs/config.yaml with zero discrepancies:

- **Base Model:** Official Pretrained Causal Demucs dns48 (18,867,937 parameters, hidden=48, depth=5, kernel_size=8, stride=4, resample=4, normalize=True, causal=True).
- **Audio Format:** 16,000 Hz, 16-bit PCM mono.
- **Training Split:** 10,802 paired utterances across 26 speakers -> 12,272 sliding-window chunks (segment = 4.0s / 64,000 samples, stride = 2.0s / 32,000 samples, batch_size = 16 -> 767 batches/epoch).
- **Validation Split:** 770 full-length uncropped utterances (p226: 353, p287: 417).
- **Test Split:** 824 full-length utterances (p232, p257) — 100% excluded from Phase 3B.
- **Loss Function (Weights Applied Once):** `L_total = 1.0 * L1(y, y_hat) + 0.1 * L_SC(y, y_hat) + 0.1 * L_Mag(y, y_hat) with fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240].`
- **Optimizer & Schedule:** Adam (lr = 5e-5, betas = (0.9, 0.999), weight_decay = 0.0), gradient clipping max_norm = 5.0, epochs = 15, seed = 42.

## 3. Preflight Test Results (job_019_phase3b_preflight)

Logged in results/preflight_report.json prior to launching the 15-epoch run:

- **Runtime Restoration:** Automatically detected missing /content/data_16k on the fresh Colab runtime and restored all 24,792 16 kHz .wav files from /content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar in ~14s.
- **Speaker Isolation Assertion:** Verified 0 overlap between training (26 speakers), validation (p226, p287), and test (p232, p257).
- **10-Batch Mini Training Check (batch_size=16, shape [16, 1, 64000]):** Finite loss across all 10 batches (mean_total_loss = 0.08879, l1 = 0.00748, raw_sc = 0.19450, raw_mag = 0.61860); finite gradients on all 48 parameter tensors (grad_norm clipped at 5.0); confirmed non-zero parameter updates (param_l2_diff = 0.02564 > 0); confirmed atomic checkpoint save and reload equivalence (max_abs_diff = 0.00e+00).

## 4. GPU Profiling Results (NVIDIA L4, 24 GB VRAM)

- **Peak GPU Memory Allocated:** 4,189.64 MB (~4.19 GB, 17.5% of L4 capacity; completely stable across all 15 epochs).
- **Average Batch Processing Time:** 282.4 ms/batch (~3.54 batches/sec).
- **Measured Single Epoch Duration:** ~270.0 seconds (~231s for 767 training batches + ~39s for 770 validation utterances including parallel Wideband PESQ and STOI).

## 5. Total Training Duration

- **Full 15-Epoch Run (job_020_phase3b_full_training, including Epoch 00 zero-shot baseline validation):** 4,123.87 seconds (68.73 minutes), completed with exit code 0 and 0 failed batches or validation utterances.

## 6. Epoch-by-Epoch Training & Validation Metrics (results/training_history.csv)

| Ep | Train Loss | Train L1 | Train SC | Train Mag | Val Loss | Val L1 | Val SC | Val Mag | Val PESQ (Mean ± Std) | Val PESQ Med | Val STOI (Mean ± Std) | Val STOI Med | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 00 | — | — | — | — | 0.1342 | 0.0092 | 0.1801 | 1.0703 | 2.0691 ± 0.6258 | 2.0111 | 0.8523 ± 0.0964 | 0.8691 | 40.5 |
| 01 | 0.0758 | 0.0053 | 0.1658 | 0.5389 | 0.0965 | 0.0075 | 0.1255 | 0.7650 | 2.4823 ± 0.6448 | 2.4884 | 0.8759 ± 0.0802 | 0.8877 | 268.8 |
| 02 | 0.0705 | 0.0050 | 0.1467 | 0.5080 | 0.0950 | 0.0073 | 0.1235 | 0.7530 | 2.5078 ± 0.6469 | 2.5141 | 0.8786 ± 0.0788 | 0.8911 | 269.8 |
| 03 | 0.0683 | 0.0049 | 0.1377 | 0.4963 | 0.0942 | 0.0073 | 0.1225 | 0.7463 | 2.5295 ± 0.6472 | 2.5456 | 0.8798 ± 0.0776 | 0.8930 | 269.7 |
| 04 | 0.0668 | 0.0048 | 0.1324 | 0.4871 | 0.0933 | 0.0073 | 0.1210 | 0.7389 | 2.5259 ± 0.6353 | 2.5320 | 0.8818 ± 0.0766 | 0.8953 | 269.7 |
| 05 | 0.0656 | 0.0048 | 0.1291 | 0.4791 | 0.0930 | 0.0073 | 0.1213 | 0.7355 | 2.5322 ± 0.6457 | 2.5252 | 0.8817 ± 0.0770 | 0.8945 | 270.1 |
| 06 | 0.0647 | 0.0047 | 0.1270 | 0.4731 | 0.0925 | 0.0073 | 0.1213 | 0.7308 | 2.5523 ± 0.6514 | 2.5530 | 0.8825 ± 0.0773 | 0.8955 | 269.9 |
| 07 | 0.0639 | 0.0047 | 0.1243 | 0.4676 | 0.0927 | 0.0073 | 0.1221 | 0.7314 | 2.5486 ± 0.6503 | 2.5467 | 0.8823 ± 0.0772 | 0.8962 | 269.9 |
| 08 | 0.0632 | 0.0047 | 0.1227 | 0.4626 | 0.0921 | 0.0073 | 0.1213 | 0.7273 | 2.5567 ± 0.6548 | 2.5696 | 0.8834 ± 0.0774 | 0.8949 | 269.9 |
| 09 | 0.0625 | 0.0046 | 0.1213 | 0.4574 | 0.0919 | 0.0072 | 0.1201 | 0.7271 | 2.5614 ± 0.6629 | 2.5648 | 0.8844 ± 0.0764 | 0.8989 | 270.1 |
| 10 | 0.0618 | 0.0046 | 0.1198 | 0.4522 | 0.0923 | 0.0073 | 0.1217 | 0.7290 | 2.5634 ± 0.6626 | 2.5633 | 0.8832 ± 0.0772 | 0.8960 | 270.0 |
| 11 | 0.0612 | 0.0046 | 0.1186 | 0.4475 | 0.0924 | 0.0073 | 0.1219 | 0.7284 | 2.5572 ± 0.6616 | 2.5682 | 0.8839 ± 0.0774 | 0.8974 | 269.9 |
| 12 | 0.0607 | 0.0046 | 0.1174 | 0.4438 | 0.0923 | 0.0073 | 0.1217 | 0.7289 | 2.5556 ± 0.6662 | 2.5661 | 0.8835 ± 0.0779 | 0.8978 | 270.3 |
| 13 | 0.0601 | 0.0045 | 0.1165 | 0.4388 | 0.0920 | 0.0072 | 0.1213 | 0.7267 | 2.5528 ± 0.6617 | 2.5616 | 0.8839 ± 0.0778 | 0.8982 | 270.1 |
| 14 | 0.0594 | 0.0045 | 0.1157 | 0.4336 | 0.0920 | 0.0073 | 0.1215 | 0.7263 | 2.5691 ± 0.6668 | 2.5843 | 0.8835 ± 0.0784 | 0.8981 | 270.1 |
| 15 | 0.0590 | 0.0045 | 0.1148 | 0.4305 | 0.0924 | 0.0073 | 0.1223 | 0.7290 | 2.5646 ± 0.6749 | 2.5697 | 0.8831 ± 0.0791 | 0.8966 | 270.2 |

## 7. Best Validation-Loss Epoch

- **Epoch 09 (best_val_loss.th):** Validation Total Loss = 0.09193 (L1 = 0.00721, Raw SC = 0.12008, Raw Mag = 0.72711 — a 31.5% reduction from the Epoch 00 zero-shot validation loss of 0.13422). Validation PESQ = 2.5614 ± 0.6629 (Median: 2.5648). Validation STOI = 0.8844 ± 0.0764 (Median: 0.8989).

## 8. Best Validation-PESQ Epoch (Selected Primary Checkpoint)

- **Epoch 14 (best_val_pesq.th):** Validation Wideband PESQ = 2.5691 ± 0.6668 (Median: 2.5843). Validation STOI = 0.8835 ± 0.0784 (Median: 0.8981). Validation Total Loss = 0.09203 (L1 = 0.00725, Raw SC = 0.12152, Raw Mag = 0.72628).

## 9. Final Validation Comparison Against Phase 3A Baselines (770 Validation Utterances)

| Condition / Model | Val Loss | Val PESQ (Mean ± Std) | Val PESQ Med | Val STOI (Mean ± Std) | Val STOI Med | Delta PESQ vs Noisy | Delta PESQ vs Zero-Shot | Delta STOI vs Noisy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Noisy Input (Unprocessed) | — | 1.5714 ± 0.5350 | 1.4186 | 0.8126 ± 0.1118 | 0.8286 | — | -0.4977 | — |
| 2. Stationary Spectral Gating | — | 1.2744 ± 0.1960 | 1.2239 | 0.7392 ± 0.0990 | 0.7589 | -0.2970 | -0.7947 | -0.0734 |
| 3. Pretrained Demucs dns48 (Zero-Shot, Ep 00) | 0.1342 | 2.0691 ± 0.6258 | 2.0111 | 0.8523 ± 0.0964 | 0.8691 | +0.4977 | — | +0.0397 |
| 4a. Fine-Tuned Demucs dns48 (best_val_loss.th, Ep 09) | 0.0919 | 2.5614 ± 0.6629 | 2.5648 | 0.8844 ± 0.0764 | 0.8989 | +0.9900 | +0.4923 | +0.0718 |
| 4b. Fine-Tuned Demucs dns48 (best_val_pesq.th, Ep 14) | 0.0920 | 2.5691 ± 0.6668 | 2.5843 | 0.8835 ± 0.0784 | 0.8981 | +0.9977 | +0.5000 | +0.0709 |

## 10. Persistent Checkpoint Locations on Google Drive

All three checkpoints were saved atomically to persistent Google Drive storage (/content/drive/MyDrive/Speech-Enhancement/checkpoints/), each containing model_state_dict, optimizer_state_dict, epoch, best_val_loss, best_val_pesq, best_val_stoi, config, seed, and history:

1. `/content/drive/MyDrive/Speech-Enhancement/checkpoints/best_val_pesq.th:` 226,478,148 bytes (226.48 MB — Epoch 14, Primary Selected Checkpoint)
2. `/content/drive/MyDrive/Speech-Enhancement/checkpoints/best_val_loss.th:` 226,476,868 bytes (226.48 MB — Epoch 09)
3. `/content/drive/MyDrive/Speech-Enhancement/checkpoints/checkpoint.th:` 226,477,767 bytes (226.48 MB — Epoch 15 latest resumable state)

## 11. Warnings, Failures, or Deviations

- **Failures / NaNs / Dropped Batches:** 0 across all 15 epochs (11,505 training gradient steps and 12,320 validation utterance evaluations).
- **Hyperparameter Deviations:** 0 — all initial experimental hyperparameters (epochs=15, lr=5e-5, batch_size=16, segment=4.0s, stride=2.0s, grad_clip=5.0) ran cleanly without alteration.
- **Test-Set Isolation:** Confirmed — the 824 official test utterances were never touched during Phase 3B.
