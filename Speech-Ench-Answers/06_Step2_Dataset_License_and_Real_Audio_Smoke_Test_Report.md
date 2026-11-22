> **Document ID:** `06` | **Phase:** Step 2 — Dataset License & Real-Audio Smoke Test | **Role:** Implementation & Verification  
> **Source Word Document:** `Step2_Dataset_License_and_Real_Audio_Smoke_Test_Report.docx` | **Repository Path:** `Speech-Ench-Answers/06_Step2_Dataset_License_and_Real_Audio_Smoke_Test_Report.md`

---
# Step 2 Technical Report: Dataset Verification, License Audit & Stage 2 Real-Audio Smoke Test

**Audited By:** Research / QA Engineer Subagent (qa-research-engineer, Model: Pro, Workspace: inherit, Permissions: Read-only)

## 1. Dataset Identity & Official Documentation Verification

Verified directly from the official University of Edinburgh DataShare repository (`https://datashare.ed.ac.uk/handle/10283/2791`, `DOI: 10.7488/ds/2117`):

- **Official Title:** *Noisy speech database for training speech enhancement algorithms and TTS models* (Valentini-Botinhao et al., 2016).
- **Official Documentation (logfiles.zip & testset_txt.zip):**
  - `log_readme.txt (verbatim, 3 lines):`

`The log files describe the noisy condition per wavfile for each of the datasets as follows:`

`wavfile noise SNR(dB)`

  - `log_trainset_28spk.txt:` Confirmed 11,572 lines (wavfile noise SNR(dB) entries at 0, 5, 10, and 15 dB SNR).
  - `log_testset.txt:` Confirmed 824 lines (p232 and p257 speakers across bus, cafe, living, office, psquare noises at 2.5, 7.5, 12.5, and 17.5 dB SNR).
  - `p232_001.txt (verbatim transcript):` "Please call Stella."

## 2. Item-Level License Findings & Restrictions

Retrieved verbatim from `https://datashare.ed.ac.uk/bitstream/handle/10283/2791/license_text` and saved to `bridge/verified_license_text.txt`:

- **Exact License: Creative Commons Attribution 4.0 International Public License (CC BY 4.0)** (`http://creativecommons.org/licenses/by/4.0/legalcode`).
- **Granted Rights (Section 2.a & Section 4):** Worldwide, royalty-free, irrevocable license to extract, reuse, reproduce, and share all or a substantial portion of the database contents, and to produce and share adapted material (including downsampled 16 kHz audio and trained models).
- **Conditions & Restrictions (Section 3.a):**
1. **Attribution:** Must credit the creator (Cassia Valentini-Botinhao, University of Edinburgh), retain copyright/warranty notices, and link to DOI: 10.7488/ds/2117 and the CC BY 4.0 license.
2. **Modification Notice:** Must state that the original 48 kHz .wav files were resampled to 16 kHz mono.
3. **No Downstream Restrictions:** Cannot apply additional legal or technological restrictions that prevent recipients from exercising CC BY 4.0 rights on the dataset.

## 3. Executed Commands, Outputs & Error Summary (Google Colab L4 GPU)

| Job ID | Executed Command | Return Code | Elapsed | Errors | Verbatim Output Summary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| job_002 | python bridge/step2_fetch_docs.py | 0 | 14.40s | 0 | Fetched verbatim license_text (CC BY 4.0) and testset_txt.zip (824 transcript files). |
| job_003 | python bridge/step2_run_real_pair.py | 0 | 71.16s | 0 | Extracted single pair p232_001.wav (167,208 bytes each), deleted temporary test zip archives, and ran python scripts/04_smoke_test.py --clean_wav /content/stage2_verify/clean_p232_001.wav --noisy_wav /content/stage2_verify/noisy_p232_001.wav. |
| job_004 | python bridge/step2_parse_metadata.py | 0 | 5.31s | 0 | Fetched logfiles.zip, inspected log_readme.txt, log_testset.txt (824 lines), and log_trainset_28spk.txt (11,572 lines). |

### Verbatim Stage 2 Console Output from scripts/04_smoke_test.py

`========================================================================`

`[STAGE 2] REAL AUDIO PAIR & METRICS (PESQ / STOI) SMOKE TEST`

`========================================================================`

`Resampling real pair from 48000 Hz -> 16000 Hz...`

`Real Pair (clean_p232_001.wav, N=27861 samples @ 16000 Hz):`

  `Noisy      -> PESQ(wb): 2.9225 | STOI: 0.8966`

  `dns48 Zero -> PESQ(wb): 3.0272 | STOI: 0.9015`

`ALL SMOKE TEST CHECKS PASSED SUCCESSFULLY.`

## 4. Real-Audio Test Details & 16 kHz PESQ / STOI Results

- **Single Utterance Tested:** `p232_001.wav` ("Please call Stella.")
- **Original Format:** 48,000 Hz, 16-bit PCM mono (167,208 bytes; 83,582 samples at 48 kHz).
- **Resampled Format:** 16,000 Hz, float32 mono tensor (N = 27,861 samples = 1.7413 seconds at 16 kHz), strictly aligned (`clean_wav.shape == noisy_wav.shape == enhanced_wav.shape == [1, 27861]`).
- **Metric Results at 16 kHz:**

| Condition | Wideband PESQ (16 kHz, "wb") | STOI (16 kHz, extended=False) | Improvement vs. Noisy |
| :--- | :--- | :--- | :--- |
| Unprocessed Noisy (p232_001.wav) | 2.9225 | 0.8966 | — |
| Pretrained Demucs dns48 (Zero-Shot) | 3.0272 | 0.9015 | +0.1047 PESQ / +0.0049 STOI |

*(Note: As required, this single-utterance test verifies end-to-end real-audio loading, 48k -> 16k resampling, alignment, model inference, and metric computation; full statistical performance will be evaluated across all 824 test utterances).*

## 5. QA / Research Engineer Recommendation: GO

- **Decision: GO for Full Dataset Preparation.**
- **Strict Restrictions Confirmed:**
  - Full training dataset was NOT downloaded (clean_trainset_28spk_wav.zip and noisy_trainset_28spk_wav.zip were untouched).
  - Full dataset was NOT resampled or preprocessed.
  - Model training/fine-tuning was NOT started.
  - Project files (configs/config.yaml, scripts/04_smoke_test.py, src/) were NOT modified.
