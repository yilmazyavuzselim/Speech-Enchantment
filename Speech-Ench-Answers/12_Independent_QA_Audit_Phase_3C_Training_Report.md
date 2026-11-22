> **Document ID:** `12` | **Phase:** Phase 3C — Independent Training QA Audit | **Role:** Independent QA Audit (Read-Only)  
> **Source Word Document:** `Independent_QA_Audit_Phase_3C_Training_Report.docx` | **Repository Path:** `Speech-Ench-Answers/12_Independent_QA_Audit_Phase_3C_Training_Report.md`

---
# Independent Technical QA Audit Report: Phase 3C (Training Implementation & Checkpoint Verification)

**Audited By:** Research / QA Engineer Subagent (qa-research-engineer, Model: Pro, Workspace: inherit, Write access: DISABLED, Strictly read-only)

## Executive Summary & Final Verdict

- **Final Verdict: APPROVED (Phase 4 Can Begin)**
- **Critical Issues:** 0
- **Major Issues:** 0
- **Minor Observations:** 1 (Informational only; documented in Section 4)

All Phase 3B training code, loss formulations, gradient clipping mechanics, reproducibility seedings, validation protocols, and persistent Google Drive checkpoints were independently audited against actual source files, runtime logs, and Colab checkpoint inspections. The official test set (test_manifest.json, 824 utterances) remained 100% untouched throughout Phase 3B.

## 1. Files Inspected

1. `src/training/trainer.py (Lines 1–648)`
2. `scripts/06_finetune_demucs.py (Lines 1–128)`
3. `src/training/stft_loss.py (Lines 1–140)`
4. `src/models/demucs_loader.py (Lines 1–145)`
5. `src/data/dataset.py (Lines 1–278)`
6. `configs/config.yaml (Lines 1–80)`
7. `results/preflight_report.json`
8. `results/training_history.csv (17 lines: header + Epoch 0 + Epochs 1–15)`
9. `results/training_summary.json`
10. `results/training_curves.png (Visually inspected 3-panel plot)`
11. `results/phase3c_checkpoint_audit_evidence.json (Colab runtime inspection of Google Drive .th files)`
12. `results/test_baseline_summary.json & results/test_baseline_per_utterance.csv`

## 2. Checks Performed

| Audit Area | Verification Checks Performed | Status | Primary Evidence |
| :--- | :--- | :--- | :--- |
| 1A. Config Audit | dns48, 16,000 Hz mono, lr=5e-5, batch_size=16, 4.0s seg / 2.0s stride, clip=5.0, 15 epochs | PASS | config.yaml, trainer.py (L64–L82), training_summary.json |
| 1B. Training Loop | model.train(), finite grads check, clip_grad_norm_(5.0), parameter updates, deterministic seeds | PASS | trainer.py (L316–L363), preflight_report.json |
| 1C. Validation Loop | model.eval(), torch.no_grad(), 770 full uncropped utterances, 16 kHz "wb" PESQ & STOI | PASS | trainer.py (L229–L314) |
| 2. Loss Function | total_loss = l1_loss + 0.1 * raw_sc_loss + 0.1 * raw_mag_loss; weights applied once; exact CSV consistency | PASS | stft_loss.py (L125–L140), trainer.py (L140–L153), training_history.csv |
| 3. Checkpoints | best_val_pesq.th (Ep 14), best_val_loss.th (Ep 9), checkpoint.th (Ep 15) on Drive + inference test | PASS | phase3c_checkpoint_audit_evidence.json |
| 4. History & Plots | 15 completed epochs + Ep 0 baseline; 0 NaNs; 68.73 min runtime; visual curve verification | PASS | training_history.csv, training_curves.png |
| 5. Val Performance | Ep 0 (2.0691) -> Ep 9 (2.5614) -> Ep 14 (2.5691); convergence & stability analysis | PASS | training_summary.json |
| 6. Test Protection | Zero test manifest/speaker access in trainer; Phase 3A test files unmodified | PASS | trainer.py (L102–L138), test_baseline_summary.json |

## 3. Evidence from Actual Files and Runtime

### A. Training Implementation & Configuration Audit (src/training/trainer.py & scripts/06_finetune_demucs.py)

- **Configuration Parity:** Inspection of DemucsTrainer.__init__ (L46–L99) and results/training_summary.json confirms zero differences between the approved configuration and the actual executed configuration: Pretrained model dns48 (18,867,937 parameters, sample_rate=16000, chin=1, chout=1, hidden=48, causal=True); lr=5e-5, betas=(0.9, 0.999), weight_decay=0.0, batch_size=16, epochs=15, grad_clip_max_norm=5.0, segment_seconds=4.0 (64,000 samples), stride_seconds=2.0 (32,000 samples).
- **Training Loop (train_one_epoch, L316–L379):** self.model.train() is explicitly called at Line 321. Deterministic seeding is enforced via set_global_seed(self.seed) (Line 54) and a per-epoch seeded DataLoader (epoch_seed = self.seed + epoch at Lines 325–334). Every batch verifies torch.isfinite(est).all() (Line 342), torch.isfinite(total_loss) (Line 152), and torch.isfinite(p.grad).all() across all parameters (Lines 350–353). Gradient clipping torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0) is applied at Lines 355–357 before self.optimizer.step() (Line 361).
- **Validation Loop (validate, L229–L314):** self.model.eval() (Line 231) and with torch.no_grad(): (Line 243) are enforced. Evaluates all 770 full-length uncropped validation utterances (val_manifest.json), asserting est.shape == clean.shape (Line 249) and len(eval_tasks) == 770 (Line 273). Calculates Wideband PESQ (16000, "wb") and STOI (16000, extended=False) via _compute_single_utterance_metrics.

### B. Loss Function Audit (src/training/stft_loss.py & src/training/trainer.py)

- **Single Weight Application:** MultiResolutionSTFTLoss.forward() (L125–L140) returns the raw unweighted mean (sc_loss, mag_loss) across the 3 resolutions ([1024, 2048, 512]). DemucsTrainer.compute_loss() (L140–L153) applies the weights exactly once: total_loss = 1.0 * l1_loss + 0.1 * raw_sc_loss + 0.1 * raw_mag_loss.
- **Empirical Verification Across training_history.csv:** Every row in results/training_history.csv satisfies loss = l1 + 0.1 * sc + 0.1 * mag to exact 6-decimal precision:
  - `Epoch 0 (Val): 0.009184 + 0.1(0.180113) + 0.1(1.070264) = 0.134222 (logged: 0.134222)`
  - `Epoch 9 (Val): 0.007206 + 0.1(0.120084) + 0.1(0.726904) = 0.091905 (logged: 0.091905)`
  - `Epoch 14 (Train): 0.004505 + 0.1(0.115737) + 0.1(0.433593) = 0.059438 (logged: 0.059438)`
  - `Epoch 14 (Val): 0.007253 + 0.1(0.121515) + 0.1(0.726281) = 0.092033 (logged: 0.092033)`

## 4. Checkpoint Integrity Findings

Independently verified via Google Drive API (gdrive readonly ls 1IFO3dEWHdxE1PaYtnSCkwJ7EIDJpiY27) and Colab runtime inspection (job_022_phase3c_checkpoint_audit, logged in results/phase3c_checkpoint_audit_evidence.json):

| Checkpoint File | Google Drive ID | Exact Size (Bytes) | Saved Epoch | Model Tensors | Optimizer State | Stored best_val_loss (Ep) | Stored best_val_pesq (Ep) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| best_val_pesq.th | 1e0VGfFb4exie8aqBTyvV9Gq0Ey3W4jlj | 226,478,148 | Epoch 14 | 48 | True | 0.091905 (Ep 9) | 2.5691 (Ep 14) |
| best_val_loss.th | 1vj8b1hvuBuCpYAue9oulDiUMVzGkmsVJ | 226,476,868 | Epoch 9 | 48 | True | 0.091905 (Ep 9) | 2.5614 (Ep 9) |
| checkpoint.th | 1lg8HMMYcJxc4w09mQ8eIS4Ugg9JVu4d_ | 226,477,767 | Epoch 15 | 48 | True | 0.091905 (Ep 9) | 2.5691 (Ep 14) |

- **Independent Reload & Forward-Pass Check on Validation Sample p226_001.wav:** In job_022_phase3c_checkpoint_audit, best_val_pesq.th was loaded into a fresh Demucs model instance on Colab and evaluated on validation utterance p226_001.wav (N = 36,480 samples @ 16 kHz): Output shape [1, 36480], all_finite: true. Wideband PESQ on p226_001 improved from 1.3900 (Noisy) to 2.2743 (best_val_pesq.th) (+0.8843 PESQ), and STOI improved from 0.7481 (Noisy) to 0.8227 (+0.0746 STOI).
- **Minor Observation 1 (Informational):** In trainer.py (Lines 457–464), the .th file extra dictionary stores best_val_loss, best_val_loss_epoch, best_val_pesq, best_val_pesq_epoch, best_val_stoi, and latest_metrics (while the full config dict and seed: 42 are stored in the companion persistent file results/training_summary.json rather than duplicated inside extra). All three checkpoints load cleanly and deterministically via load_checkpoint().

## 5. Training History Verification

- `results/training_history.csv:` Verified 17 lines (header + Epoch 0 zero-shot baseline + Epochs 1–15). Zero missing or non-finite values.
- **Duration & Memory Verification:** Sum of Epoch 0 (40.48s) + Epochs 1–15 (15 x ~270.0s = 4,049.55s) plus checkpoint writes equals 4,123.87s (68.73 minutes). Peak GPU memory is 4,189.64 MB across all 15 epochs.
- `results/training_curves.png:` Visually inspected; accurately plots Training/Validation Loss, Validation Wideband PESQ, and Validation STOI against the Phase 3A reference lines.

## 6. Validation Metric Verification & Epoch 14 vs. Epoch 9 Analysis

| Validation Milestone (770 Utterances, Speakers p226 & p287) | Val Total Loss | Val PESQ (Mean ± Std) | Val PESQ Median | Val STOI (Mean ± Std) | Val STOI Median |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Noisy Baseline (Phase 3A) | — | 1.5714 ± 0.5350 | 1.4186 | 0.8126 ± 0.1118 | 0.8286 |
| Pretrained dns48 Zero-Shot (Epoch 0) | 0.134222 | 2.0691 ± 0.6258 | 2.0111 | 0.8523 ± 0.0964 | 0.8691 |
| Fine-Tuned Epoch 9 (best_val_loss.th) | 0.091905 | 2.5614 ± 0.6629 | 2.5648 | 0.8844 ± 0.0764 | 0.8989 |
| Fine-Tuned Epoch 14 (best_val_pesq.th) | 0.092033 | 2.5691 ± 0.6668 | 2.5843 | 0.8835 ± 0.0784 | 0.8981 |
| Fine-Tuned Epoch 15 (checkpoint.th) | 0.092425 | 2.5646 ± 0.6749 | 2.5697 | 0.8831 ± 0.0791 | 0.8966 |

- **Primary Checkpoint Justification (best_val_pesq.th, Epoch 14):** Epoch 14 achieves both the highest validation mean PESQ (2.5691, +0.5000 over zero-shot dns48) and the highest validation median PESQ (2.5843, +0.5732 over zero-shot dns48), with negligible difference in validation loss (0.092033 vs. 0.091905 at Epoch 9) and STOI (0.8835 vs. 0.8844 at Epoch 9).
- **Convergence Stability vs. Overfitting (Epochs 9–15):** Across Epochs 9–15, validation loss remains tightly bounded in [0.0919, 0.0924], validation PESQ in [2.5528, 2.5691], and validation STOI in [0.8831, 0.8844]. This demonstrates a stable asymptotic convergence plateau rather than metric instability or overfitting.

## 7. Test-Set Integrity

- _verify_dataset_isolation() (src/training/trainer.py, L102–L138) programmatically asserts that test speakers (p232, p257) are disjoint from training and validation sets.
- test_manifest.json is never referenced anywhere in src/training/trainer.py or scripts/06_finetune_demucs.py.
- results/test_baseline_summary.json and results/test_baseline_per_utterance.csv remain completely unmodified from Phase 3A.

## 8. Final Recommendation

### VERDICT: APPROVED — Phase 4 (Official Test-Set Evaluation) Can Begin.

All Phase 3B training deliverables and persistent checkpoints have passed independent QA verification with zero critical or major issues.
