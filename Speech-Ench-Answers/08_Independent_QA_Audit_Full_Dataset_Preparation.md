> **Document ID:** `08` | **Phase:** Phase 2B — Independent QA Audit of Dataset Preparation | **Role:** Independent QA Audit (Read-Only)  
> **Source Word Document:** `Independent_QA_Audit_Full_Dataset_Preparation.docx` | **Repository Path:** `Speech-Ench-Answers/08_Independent_QA_Audit_Full_Dataset_Preparation.md`

---
# Independent Technical QA Audit Report: Full Dataset Preparation

**Audited By:** Research / QA Engineer Subagent (qa-research-engineer, Model: Pro, Workspace: inherit, Permissions: Strictly read-only)

## 1. Master Audit Matrix (PASS / WARNING / FAIL)

| Audit # | Audit Item | Status | Independent Verification Evidence |
| :--- | :--- | :--- | :--- |
| 1.1 | Existence & integrity of original 48 kHz archives on Google Drive | PASS | Independently verified via gdrive readonly ls 1jal2HDYFxYvb6QbwkzNq1RKYih4rJTjC (/MyDrive/Speech-Enhancement/data/raw_48k_zips) and results/download_report.json: all 8 official files present (5.26 GB), exact byte counts matched, SHA-256 recorded, and zipfile.testzip() == None. |
| 1.2 | Existence & integrity of edinburgh_16k.tar on Google Drive | PASS | Independently verified via gdrive readonly info 10WnUqxish9xZ1P5Wbt9gAEfF1stP3HEJ (/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar, 2,367,969,280 bytes / 2.205 GB). |
| 1.3 | Availability of extracted 16 kHz dataset on Colab | PASS | Currently extracted and live at /content/data_16k/ (24,792 16 kHz mono .wav files) on the active Colab L4 runtime. |
| 1.4 | Manifest path structure (clean_path / noisy_path) | PASS | Inspected train_manifest.csv, val_manifest.csv, and test_manifest.csv: each entry stores both the fast local Colab SSD path (clean_path: /content/data_16k/...) and a portable relative path (clean_rel_path: clean_trainset_28spk_wav/p226_001.wav). |
| 1.5 | Restoration after a Colab runtime reset | WARNING | Because /content/data_16k is on temporary Colab SSD (for 20x–50x faster DataLoader I/O), a Colab runtime reset requires unpacking edinburgh_16k.tar (tar -xf /content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar -C /content, ~14s). Our upcoming src/data/dataset.py / training script must auto-run this unpack step if /content/data_16k is missing. |
| 2.1 | Exact utterance counts (Train: 10,802, Val: 770, Test: 824) | PASS | Independently verified across CSV/JSON manifests and results/dataset_validation_report.json: 10,802 + 770 = 11,572 train archive pairs; 824 test archive pairs (12,396 total pairs). |
| 2.2 | Speaker IDs & mutual speaker isolation (Train ∩ Val ∩ Test = ∅) | PASS | Train: 26 speakers; Val: 2 speakers (p226: 353, p287: 417); Test: 2 speakers (p232: 393, p257: 431). All three speaker sets are mutually disjoint (overlap: []). |
| 2.3 | Clean/noisy pair matching, sample-count equality, SR (16000), mono (1 ch), no NaN/Inf, no duplicates | PASS | Verified in validate_single_audio_pair (src/data/splits.py) and manifest outputs: 0 missing files, 0 duration/sample mismatches, 0 non-16kHz/non-mono files, 0 duplicates, 0 NaN/Inf values. |
| 3.1 | Preprocessing & split code review (download.py, preprocess.py, splits.py, 01–03 scripts) | PASS | Line-by-line review confirmed atomic .part / .tmp file writes, streaming SHA-256, torchaudio.transforms.Resample(48000, 16000), strict error raising (zero silent sample dropping), and deterministic sorting. |
| 4.1 | Direct compatibility with dns48 training pipeline | WARNING (Prerequisite) | Manifests and 16 kHz audio are 100% compatible with dns48, but the training data loader (src/data/dataset.py), baseline (src/baselines/spectral_gating.py), trainer (src/training/trainer.py), and evaluator (src/evaluation/evaluator.py) have not been created yet (per our phased implementation order). |

## 2. Independently Verified Dataset Counts

| Split | Manifest File | Verified Utterance Pairs | Speaker Count & IDs | Total Duration | Noise Conditions | SNR Levels (dB) | Anomalies |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Train | train_manifest.json / .csv | 10,802 | 26 speakers (p227–p286, excl. p226, p287) | 8.758 hrs (31,529.0s) | 10 training noises (~1,080/noise) | 0, 5, 10, 15 | 0 |
| Validation | val_manifest.json / .csv | 770 | 2 speakers (p226: 353, p287: 417) | 0.633 hrs (2,279.6s) | 10 training noises (~77/noise) | 0, 5, 10, 15 | 0 |
| Test | test_manifest.json / .csv | 824 | 2 speakers (p232: 393, p257: 431) | 0.576 hrs (2,072.1s) | 5 unseen noises (bus, cafe, living, office, psquare) | 2.5, 7.5, 12.5, 17.5 | 0 |
| Total | All 3 Splits | 12,396 (24,792 WAVs) | 30 mutually disjoint speakers | 9.967 hrs (35,880.7s) | 15 total noise types | Balanced | 0 |

## 3. Manifest Path & Persistence Assessment

1. **Where Manifest Paths Point:** Each manifest entry contains both: (a) `clean_path / noisy_path` (`/content/data_16k/clean_trainset_28spk_wav/p227_001.wav`, pointing to Colab local NVMe SSD for fast training I/O), and (b) `clean_rel_path / noisy_rel_path` (`clean_trainset_28spk_wav/p227_001.wav`, portable path relative to any data_16k root).
2. **Persistence Across Colab Runtime Resets:** The 2.205 GB archive `/content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar` is stored permanently on your Google Drive (ID: 10WnUqxish9xZ1P5Wbt9gAEfF1stP3HEJ) with internal archive root data_16k/. Whenever a new Colab runtime starts, running `tar -xf /content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar -C /content` restores /content/data_16k/... in ~14 seconds so every clean_path and noisy_path in the manifests resolves immediately.

## 4. Preprocessing Code Review Findings

- `src/data/download.py:` Uses atomic .part downloads (L104–L130), pre-download shutil.disk_usage verification (L71–L84), and zipfile.testzip() CRC verification (L153–L158).
- `src/data/preprocess.py:` Verifies 48,000 Hz mono source properties (L46–L56), resamples via torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000) (L61–L64), saves as 16-bit PCM (subtype="PCM_16"), and raises a RuntimeError if even a single file fails (L132–L134)—guaranteeing zero silent sample dropping.
- `src/data/splits.py:` Deterministically sorts entries, verifies 1-to-1 set equality between log entries, clean WAVs, and noisy WAVs (L110–L126), checks sample-level frame parity and np.isfinite on every pair (L58–L83), and enforces disjoint speaker sets (L171–L175).

## 5. Training Pipeline Compatibility & Required Prerequisites Before Training

Because we strictly followed the staged implementation order (creating only the Step 1 smoke-test files and Step 2 dataset preparation scripts so far), the following components must be implemented before model training can begin:

| Priority | Required Component | Purpose |
| :--- | :--- | :--- |
| P1 (Required) | Auto-Restore Helper + PyTorch Dataset (src/data/dataset.py) | 1. Automatically unpack /content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar to /content/data_16k if /content/data_16k is absent after a Colab reset.
2. Resolve clean_rel_path / noisy_rel_path against config["paths"]["local_data_16k_dir"] as a fallback to clean_path.
3. Slice 4.0s (64,000-sample) training segments with 2.0s (32,000-sample) stride and padding. |
| P1 (Required) | Baseline & Zero-Shot Evaluator (src/baselines/spectral_gating.py, src/evaluation/evaluator.py, scripts/05_evaluate_baselines.py) | Evaluate Unprocessed Noisy, Stationary Spectral Gating (noisereduce), and Pretrained Demucs dns48 (Zero-Shot) across all 824 test utterances (and 770 validation utterances) before fine-tuning starts. |
| P2 (Required) | Fine-Tuning Trainer (src/training/trainer.py, scripts/06_finetune_demucs.py) | Fine-tune dns48 on train_manifest.json (10,802 pairs), validate on val_manifest.json (770 pairs) using L1 + 0.1 * SC + 0.1 * Mag loss and Wideband PESQ, and save atomic checkpoints (checkpoint.th, best_val_loss.th, best_val_pesq.th) to Google Drive. |

## 6. Final QA Recommendation: CONDITIONAL GO (Dataset 100% Approved; Build DataLoader, Baseline & Trainer Modules Next)

- **Dataset Status: 100% PASS (Zero fixes needed on the dataset, archives, or manifests).**
- **Condition for Starting Training:** Before running training, we must implement the remaining pipeline modules listed in Section 5 (src/data/dataset.py with auto-restore from edinburgh_16k.tar, src/baselines/spectral_gating.py, src/evaluation/evaluator.py, and src/training/trainer.py) and lock in our pre-training baseline/zero-shot metrics.
