> **Document ID:** `07` | **Phase:** Phase 2A — Full Dataset Preparation | **Role:** Implementation Engineer Report  
> **Source Word Document:** `Full_Dataset_Preparation_Report.docx` | **Repository Path:** `Speech-Ench-Answers/07_Full_Dataset_Preparation_Report.md`

---
# Full Dataset Preparation Report (Speech-Enhancement)

**Executed By:** Implementation Engineer Subagent (impl-engineer, Model: Pro, Workspace: inherit, Write enabled, Colab L4 GPU)

## 1. Pre-Download Storage Check & Bitstream Inspection (job_006_storage_check_v2)

Before downloading any archives, persistent Google Drive storage, Colab local NVMe SSD storage, and remote HTTP Content-Length headers were inspected via `scripts/check_storage_and_remote_sizes.py` and logged to `results/storage_check.json`:

- **Persistent Google Drive (/content/drive/MyDrive): 223.65 GB free** (of 235.68 GB total)
- **Colab Local NVMe SSD (/content): 235.42 GB free** (of 235.68 GB total)
- **Total Download Required (8 Official Bitstreams): 5,648,236,335 bytes (5.26 GB)** — over 40x storage headroom available.

## 2. Official 48 kHz Dataset Download & Preservation (scripts/01_download_dataset.py, job_009_download_dataset)

All 8 official bitstreams from `https://datashare.ed.ac.uk/bitstream/handle/10283/2791/` (DOI: 10.7488/ds/2117) were downloaded to persistent Google Drive storage (`/content/drive/MyDrive/Speech-Enhancement/data/raw_48k_zips/`), preserved intact without modification, and verified via exact byte count, SHA-256 checksum, and `zipfile.ZipFile.testzip()` in `results/download_report.json`:

| Official Bitstream | Exact Size (Bytes) | Size (MB) | SHA-256 Checksum | Zip Integrity (testzip()) |
| :--- | :--- | :--- | :--- | :--- |
| clean_trainset_28spk_wav.zip | 2,486,057,279 | 2,370.89 | dbb13b9ffe752c58764d75de272706a2b467744f33455cec6deb2b12ca01df66 | VALID (None) |
| noisy_trainset_28spk_wav.zip | 2,830,205,201 | 2,699.09 | a656af041f5a8931b02077331281c2b82ebc35b38dc83ff92935015bcd7920b4 | VALID (None) |
| clean_testset_wav.zip | 154,332,064 | 147.18 | b60bfc2b9df38466d8eb45057c087c6e5dee196c67a0ed7bca628ae2a8611c4e | VALID (None) |
| noisy_testset_wav.zip | 170,578,854 | 162.68 | ba4a893f45b627f15209ba30ab584dd495b3680cd6f216dd08d143a257611993 | VALID (None) |
| trainset_28spk_txt.zip | 6,489,312 | 6.19 | 5ec7d6ef99b78a7d039f1ecd6db7d0d5b660877b464de15e1d75de6425fbf3f8 | VALID (None) |
| testset_txt.zip | 440,277 | 0.42 | b31edb40b39c8ddc99eda13ea038f51be3065a8b70272c8051ffc2c4294cf1d3 | VALID (None) |
| logfiles.zip | 115,932 | 0.11 | ab37c41f31bc2b6c204cc10c8e5bb8bde800307fdb49f277e366d986fefc9e39 | VALID (None) |
| license_text | 17,416 | 0.02 | b34e17103bfb246f2549fc82a279e6ba28834e0cb42f76a92efc14b72e3a3723 | Plain Text |

## 3. 16 kHz Mono Dataset Preparation (scripts/02_resample_16k.py, job_010_resample_16k)

Logged in `results/resample_report.json`:

1. Extracted the 48 kHz archives to Colab local SSD (`/content/raw_48k_extracted/`) while leaving `/content/drive/MyDrive/Speech-Enhancement/data/raw_48k_zips/` untouched.
2. Verified every source .wav file is 48,000 Hz mono and downsampled all **24,792 .wav files** (11,572 clean train + 11,572 noisy train + 824 clean test + 824 noisy test) to **16,000 Hz, 16-bit PCM mono (PCM_16)** in `/content/data_16k/` with 0 errors.
3. Packaged the complete 16 kHz dataset into a persistent archive at `/content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar` (2,367,969,280 bytes / 2.205 GB) and cleaned up the temporary `/content/raw_48k_extracted/` scratch directory.

## 4. Manifest Generation, Split Verification & Data Integrity Audit (scripts/03_create_manifests_and_validate.py, job_011_manifests_and_validate)

Logged in `results/dataset_validation_report.json`:

### A. Speaker Verification & Strict Speaker Isolation

- **Configured Validation Speakers Verified:** Both `p226` (353 utterances, 1,147.2s) and `p287` (417 utterances, 1,132.4s) exist in the 28-speaker training set and total 770 utterances (2,279.6s / 0.633 hours).
- **Remaining Training Speakers (26 Speakers):** `p227, p228, p230, p231, p233, p236, p239, p243, p244, p250, p254, p256, p258, p259, p267, p268, p269, p270, p273, p274, p276, p277, p278, p279, p282, p286` total 10,802 utterances (31,529.0s / 8.758 hours).
- **Official Test Speakers (2 Unseen Speakers):** `p232` (393 utterances, 1,004.2s) and `p257` (431 utterances, 1,067.9s) total 824 utterances (2,072.1s / 0.576 hours).
- **Disjoint Speaker Check:** Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅ (is_disjoint: true).

### B. Complete Dataset Summary Table

| Metric | Train Split (train_manifest) | Validation Split (val_manifest) | Test Split (test_manifest) | Total Dataset |
| :--- | :--- | :--- | :--- | :--- |
| Utterance Pairs | 10,802 | 770 | 824 | 12,396 pairs (24,792 WAVs) |
| Speakers | 26 speakers | 2 speakers (p226, p287) | 2 speakers (p232, p257) | 30 speakers |
| Total Duration | 8.758 hours (31,529.0s) | 0.633 hours (2,279.6s) | 0.576 hours (2,072.1s) | 9.967 hours (35,880.7s) |
| Mean / Min / Max Duration | 2.919s (1.092s–15.112s) | 2.961s (1.357s–11.781s) | 2.515s (1.238s–9.769s) | 2.894s (1.092s–15.112s) |
| Sample Rate / Format | 16,000 Hz, 16-bit PCM mono | 16,000 Hz, 16-bit PCM mono | 16,000 Hz, 16-bit PCM mono | 16,000 Hz, 16-bit PCM mono |
| Noise Types | 10 (babble, cafeteria, car, kitchen, meeting, metro, restaurant, ssn, station, traffic) | Same 10 training noise types | 5 unseen (bus, cafe, living, office, psquare) | 15 total noise conditions |
| SNR Levels (dB) | 0.0 (2,700), 5.0 (2,694), 10.0 (2,708), 15.0 (2,700) | 0.0 (193), 5.0 (193), 10.0 (191), 15.0 (193) | 2.5 (207), 7.5 (206), 12.5 (207), 17.5 (204) | Balanced across all splits |

### C. Anomaly & Integrity Check Summary (Across All 12,396 Pairs)

- **Missing files:** 0
- **Corrupted audio files:** 0
- **Incorrect sample rates (!= 16000 Hz) or non-mono (!= 1 ch):** 0
- **Clean/noisy duration or sample-count mismatches:** 0
- **Duplicate utterance entries:** 0
- **NaN or Infinite audio values:** 0
- **Discarded samples:** 0 (100% of official utterances preserved)

## 5. Deliverables Created in Workspace (Speech-Enhancement)

1. **Reproducible Preprocessing Modules & CLI Scripts:** `src/data/__init__.py, src/data/download.py, src/data/preprocess.py, src/data/splits.py, scripts/check_storage_and_remote_sizes.py, scripts/01_download_dataset.py, scripts/02_resample_16k.py, scripts/03_create_manifests_and_validate.py`
2. **Dataset Manifests (utterance_id, clean_path, noisy_path, speaker_id, noise_type, snr_db, num_samples, duration_sec, sample_rate):** `manifests/train_manifest.json & .csv (10,802 pairs), manifests/val_manifest.json & .csv (770 pairs), manifests/test_manifest.json & .csv (824 pairs)`
3. **Validation & Execution Reports:** `results/storage_check.json, results/download_report.json, results/resample_report.json, results/dataset_validation_report.json`
4. **Updated Documentation:** `README.md & .gitignore`

## 6. Stopping at Phase Gate (Awaiting QA Audit & Approval)

- Model training / fine-tuning has NOT been started.
- Model architecture (src/models/demucs_loader.py) and loss functions (src/training/stft_loss.py) were NOT modified.
