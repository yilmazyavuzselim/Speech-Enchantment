> **Document ID:** `09` | **Phase:** Phase 3A — DataLoader & Baseline Evaluation | **Role:** Implementation Engineer Report  
> **Source Word Document:** `Phase_3A_DataLoader_and_Baseline_Evaluation_Report.docx` | **Repository Path:** `Speech-Ench-Answers/09_Phase_3A_DataLoader_and_Baseline_Evaluation_Report.md`

---
# Phase 3A Engineering Report: Data Loader + Baseline Evaluation

**Executed By:** Implementation Engineer Subagent (impl-engineer, Model: Pro, Workspace: inherit, Write enabled, Colab NVIDIA L4 GPU)

## 1. Architectural & Implementation Summary

### Task 1: Dataset Loader & Runtime Restoration (src/data/dataset.py)

- **Automatic Colab Runtime Restoration (ensure_local_data_16k):** Automatically checks if /content/data_16k is present; if missing (e.g., after a Colab reset), extracts the persistent Google Drive archive `/content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar` into /content in read-only mode without modifying or deleting the persistent .tar archive.
- **Portable Path Resolution:** Resolves portable relative paths (`clean_rel_path / noisy_rel_path`) against `local_data_16k_dir` (/content/data_16k) with fallback to absolute manifest paths.
- **SpeechEnhancementDataset Modes:** (a) Train Split (split="train"): Indexes 10,802 utterances into 12,272 fixed 4.0-second sliding-window chunks (64,000 samples @ 16 kHz) with 2.0-second stride (32,000 samples) and zero-padding for short/trailing windows, returning CPU float32 tensors [1, 64000] ([B, 1, 64000] when batched). (b) Validation & Test Splits (split="val", split="test"): Strict uncropped full-utterance mode returning [1, T] clean/noisy waveforms and full utterance metadata.

### Task 2: Stationary Spectral Gating Baseline (src/baselines/spectral_gating.py)

- **Reproducible Parameters:** Uses `noisereduce` with `sr=16000, stationary=True, n_fft=512, win_length=512, hop_length=128, n_std_thresh_stationary=1.5, prop_decrease=1.0`.
- **Shape & Edge-Case Safety:** Supports 1D [T], 2D [1, T], and 3D [B, 1, T] numpy arrays and PyTorch tensors, preserves exact sample length, safely handles silent/zero-variance audio (<1e-10) without divide-by-zero warnings, repairs non-finite values, and clips outputs to [-1.0, 1.0].

### Task 3: Evaluation Pipeline (src/evaluation/evaluator.py & scripts/05_evaluate_baselines.py)

- **Three Evaluated Conditions:** (1) Unprocessed Noisy Audio (noisy), (2) Stationary Spectral Gating (spectral_gating), and (3) Official Pretrained Demucs dns48 Zero-Shot (dns48_zeroshot).
- **Strict Separation & Complete Coverage:** Evaluates all 770 validation utterances and all 824 official test utterances separately (1,594 pairs x 3 conditions = 4,782 evaluations) with 0 failed samples.

## 2. Preliminary Test Suite Verification (scripts/test_phase3a_preliminary.py, job_016_preliminary_tests_v3)

Executed on Colab L4 GPU prior to running the full evaluation. All 5 preliminary verification suites passed 100%:

1. **Colab Runtime Restoration (ensure_local_data_16k):** PASSED — Verified detection of /content/data_16k and verified safe extraction from edinburgh_16k.tar into a temporary directory without altering the persistent Drive archive.
2. **Dataset & DataLoader Verification:** PASSED — 10,802 training utterances mapped to 12,272 chunks of shape (1, 64000); collated batch shape (4, 1, 64000) verified; validation p226_001 (1, 36480) and test p257_001 (1, 35513) returned uncropped.
3. **Spectral Gating Baseline Test:** PASSED — Verified 1D array, 2D tensor, and all-zero silent inputs preserve exact length and finite values.
4. **Demucs dns48 Zero-Shot Inference Test:** PASSED — Verified CUDA inference returns exact shape (1, 1, 35513) with finite values.
5. **PESQ & STOI Calculation Test:** PASSED — Verified Wideband PESQ (16 kHz, "wb") and STOI (16 kHz, extended=False) across all 3 conditions on real audio.

## 3. Full Baseline Benchmark Results (job_017_evaluate_baselines_full)

### A. Validation Set Results (770 Utterances — Speakers p226 & p287 — Used for Model Selection)

| Condition | PESQ (Mean ± Std) | PESQ Median | STOI (Mean ± Std) | STOI Median | Delta PESQ vs Noisy | Delta STOI vs Noisy | Success / Fail |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Unprocessed Noisy | 1.5714 ± 0.5350 | 1.4186 | 0.8126 ± 0.1118 | 0.8286 | — | — | 770 / 0 |
| 2. Stationary Spectral Gating | 1.2744 ± 0.1960 | 1.2239 | 0.7392 ± 0.0990 | 0.7589 | -0.2970 | -0.0734 | 770 / 0 |
| 3. Pretrained Demucs dns48 (Zero-Shot) | 2.0691 ± 0.6258 | 2.0111 | 0.8523 ± 0.0964 | 0.8691 | +0.4977 | +0.0397 | 770 / 0 |

### Validation Set Breakdown by SNR Level

| SNR Level | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs dns48 PESQ | Noisy STOI | Spectral Gating STOI | Demucs dns48 STOI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.0 dB | 193 | 1.2333 ± 0.2571 | 1.1718 ± 0.1434 | 1.6032 ± 0.4670 (+0.37) | 0.7278 ± 0.1276 | 0.6596 ± 0.1066 | 0.7897 ± 0.1166 (+0.062) |
| 5.0 dB | 193 | 1.3908 ± 0.3736 | 1.2412 ± 0.1609 | 1.8967 ± 0.5383 (+0.51) | 0.7942 ± 0.0896 | 0.7226 ± 0.0819 | 0.8412 ± 0.0815 (+0.047) |
| 10.0 dB | 191 | 1.6780 ± 0.5368 | 1.2976 ± 0.1711 | 2.2241 ± 0.5329 (+0.55) | 0.8503 ± 0.0825 | 0.7685 ± 0.0756 | 0.8793 ± 0.0719 (+0.029) |
| 15.0 dB | 193 | 1.9845 ± 0.5677 | 1.3874 ± 0.2299 | 2.5539 ± 0.5174 (+0.57) | 0.8784 ± 0.0745 | 0.8063 ± 0.0680 | 0.8995 ± 0.0688 (+0.021) |

### B. Official Test Set Results (824 Utterances — Speakers p232 & p257 — Reporting Only)

| Condition | PESQ (Mean ± Std) | PESQ Median | STOI (Mean ± Std) | STOI Median | Delta PESQ vs Noisy | Delta STOI vs Noisy | Success / Fail |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Unprocessed Noisy | 1.9719 ± 0.7532 | 1.7986 | 0.9211 ± 0.0709 | 0.9403 | — | — | 824 / 0 |
| 2. Stationary Spectral Gating | 1.4232 ± 0.2848 | 1.3502 | 0.8234 ± 0.0655 | 0.8320 | -0.5487 | -0.0977 | 824 / 0 |
| 3. Pretrained Demucs dns48 (Zero-Shot) | 2.5203 ± 0.6258 | 2.5263 | 0.9300 ± 0.0638 | 0.9476 | +0.5484 | +0.0089 | 824 / 0 |

### Official Test Set Breakdown by SNR Level

| SNR Level | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs dns48 PESQ | Noisy STOI | Spectral Gating STOI | Demucs dns48 STOI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2.5 dB | 207 | 1.4232 ± 0.4251 | 1.2810 ± 0.2183 | 1.9860 ± 0.4584 (+0.56) | 0.8683 ± 0.0864 | 0.7834 ± 0.0734 | 0.8907 ± 0.0748 (+0.022) |
| 7.5 dB | 206 | 1.7651 ± 0.5921 | 1.3968 ± 0.2501 | 2.3774 ± 0.5122 (+0.61) | 0.9188 ± 0.0590 | 0.8211 ± 0.0583 | 0.9260 ± 0.0573 (+0.007) |
| 12.5 dB | 207 | 2.1041 ± 0.6947 | 1.4446 ± 0.2574 | 2.6599 ± 0.5211 (+0.56) | 0.9403 ± 0.0501 | 0.8369 ± 0.0556 | 0.9444 ± 0.0500 (+0.004) |
| 17.5 dB | 204 | 2.6035 ± 0.7063 | 1.5722 ± 0.3240 | 3.0651 ± 0.4485 (+0.46) | 0.9574 ± 0.0464 | 0.8527 ± 0.0537 | 0.9594 ± 0.0478 (+0.002) |

### Official Test Set Breakdown by Unseen Noise Type

| Noise Type | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs dns48 PESQ | Noisy STOI | Demucs dns48 STOI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| bus | 164 | 2.4845 ± 0.7773 | 1.5220 ± 0.3188 | 2.8184 ± 0.5711 (+0.33) | 0.9519 ± 0.0472 | 0.9488 ± 0.0496 |
| cafe | 164 | 1.4919 ± 0.4302 | 1.2904 ± 0.1989 | 2.1181 ± 0.5447 (+0.63) | 0.8768 ± 0.0861 | 0.9032 ± 0.0721 (+0.026) |
| living | 166 | 1.6178 ± 0.4898 | 1.3628 ± 0.2245 | 2.3150 ± 0.5432 (+0.70) | 0.9040 ± 0.0626 | 0.9218 ± 0.0605 (+0.018) |
| office | 164 | 2.5361 ± 0.7057 | 1.5682 ± 0.3317 | 2.9006 ± 0.5518 (+0.36) | 0.9583 ± 0.0559 | 0.9498 ± 0.0622 |
| psquare | 166 | 1.7367 ± 0.5673 | 1.3738 ± 0.2237 | 2.4527 ± 0.5435 (+0.72) | 0.9145 ± 0.0621 | 0.9266 ± 0.0605 (+0.012) |

## 4. Deliverables Created in Workspace

1. `src/data/dataset.py:` Dataset loader with automatic Colab tar restoration (ensure_local_data_16k), portable relative-path resolution, 4.0s/2.0s sliding-window training chunking (12,272 chunks), and uncropped val/test evaluation loading.
2. `src/baselines/spectral_gating.py:` Stationary Spectral Gating baseline using noisereduce with 16 kHz shape/length preservation and edge-case protection.
3. `src/evaluation/evaluator.py & scripts/05_evaluate_baselines.py:` Complete baseline evaluation engine with alignment checks, transparent error reporting, and separate validation (770) and test (824) reports.
4. `scripts/test_phase3a_preliminary.py:` Preliminary verification suite testing DataLoader, runtime restoration, Spectral Gating, dns48 inference, and PESQ/STOI.
5. **Per-Utterance & Summary Reports in results/:** val_baseline_per_utterance.csv (2,310 rows), val_baseline_summary.csv & .json, test_baseline_per_utterance.csv (2,472 rows), and test_baseline_summary.csv & .json.
6. `README.md:` Updated with Phase 3A documentation and baseline benchmark tables.
