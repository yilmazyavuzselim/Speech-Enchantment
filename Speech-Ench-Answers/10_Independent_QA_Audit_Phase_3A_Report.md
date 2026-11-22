> **Document ID:** `10` | **Phase:** Phase 3A-QA — Independent QA Audit of Phase 3A | **Role:** Independent QA Audit (Read-Only)  
> **Source Word Document:** `Independent_QA_Audit_Phase_3A_Report.docx` | **Repository Path:** `Speech-Ench-Answers/10_Independent_QA_Audit_Phase_3A_Report.md`

---
# Independent Technical QA Audit Report: Phase 3A (Data Loader + Baseline Evaluation)

**Audited By:** Research / QA Engineer Subagent (qa-research-engineer, Model: Pro, Workspace: inherit, Permissions: Strictly read-only)

## 1. Files Inspected

- `src/data/dataset.py`
- `src/baselines/spectral_gating.py`
- `src/evaluation/evaluator.py`
- `scripts/05_evaluate_baselines.py`
- `scripts/test_phase3a_preliminary.py`
- `configs/config.yaml`
- `results/val_baseline_per_utterance.csv, results/val_baseline_summary.json, results/val_baseline_summary.csv`
- `results/test_baseline_per_utterance.csv, results/test_baseline_summary.json, results/test_baseline_summary.csv`
- `README.md`

## 2. Checks Performed

| Audit Domain | Verification Checks Performed | Status |
| :--- | :--- | :--- |
| 1. Dataset & DataLoader (src/data/dataset.py) | Train/val/test speaker isolation; portable relative-path resolution; 4.0s (64,000-sample) windows and 2.0s (32,000-sample) stride; tail zero-padding; no random cropping in val/test; clean/noisy alignment; non-destructive Colab runtime restoration from edinburgh_16k.tar. | PASS |
| 2. Spectral Gating (src/baselines/spectral_gating.py) | Noise profile estimation method (stationary=True, y_noise=None); STFT parameters (n_fft=512, win_length=512, hop_length=128, n_std_thresh_stationary=1.5, prop_decrease=1.0); root-cause signal-processing analysis of why full-utterance stationary gating degrades PESQ/STOI. | PASS |
| 3. Demucs dns48 Evaluation (src/evaluation/evaluator.py) | 16 kHz mono input verification; model.eval() and torch.no_grad() usage; exact output length/shape parity; finite value assertions; post-inference [-1.0, 1.0] clipping analysis. | PASS |
| 4. PESQ & STOI Calculation (src/evaluation/evaluator.py) | Wideband PESQ (16000 Hz, "wb") and STOI (16000 Hz, extended=False); pre-metric sample-alignment assertion; non-silent error handling; inclusion of all 770 validation and 824 test utterances (0 failures). | PASS |
| 5. Test Set Integrity & p232_001.wav Audit | Confirmed p232_001.wav was previously used only for Stage 2 pipeline mechanics verification; zero usage in training, tuning, or checkpoint selection; validation metrics isolated as sole reference for Phase 3B model selection. | PASS |
| 6. Results Reproducibility & CSV/JSON Parity | Verified exact row counts (2,310 val rows + header = 2,311 lines; 2,472 test rows + header = 2,473 lines), 0 missing values, 0 duplicates, and 100% mathematical match between per-utterance CSVs and summary JSON/CSVs. | PASS |

## 3. Evidence and Findings

### A. Dataset and DataLoader Verification (src/data/dataset.py)

- **Speaker Isolation & Zero Leakage:** Train (26 speakers, 10,802 utterances), Validation (p226 & p287, 770 utterances), and Test (p232 & p257, 824 utterances) load from separate JSON manifests with zero speaker or utterance overlap.
- **Relative-Path Resolution:** _resolve_path (L104–L118) resolves clean_rel_path and noisy_rel_path against local_data_16k_dir (/content/data_16k) first, falling back to absolute paths only if needed.
- **4-Second Windows, 2-Second Stride & Zero-Padding:** In train mode, segment_samples = 64,000 (4.0s) and stride_samples = 32,000 (2.0s). Short utterances and trailing windows are right-padded with zeros via F.pad(chunk, (0, pad_len)) so every training item has exact shape [1, 64000]. In val and test modes, utterances are never cropped or padded, returning native [1, T] tensors.
- **Non-Destructive Runtime Restoration:** ensure_local_data_16k (L17–L48) opens /content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar strictly in read-only mode ("r") and extracts to /content when /content/data_16k is missing, leaving the persistent Google Drive archive completely untouched. Verified in scripts/test_phase3a_preliminary.py.

### B. Spectral Gating Analysis (src/baselines/spectral_gating.py)

- **How the Noise Profile is Estimated:** nr.reduce_noise is called with stationary=True and y_noise=None (no external isolated noise-only reference clip is assumed or supplied). Consequently, noisereduce estimates the frequency-bin mean (mu_f) and standard deviation (sigma_f) across the entire noisy utterance (which contains both speech and background noise).
- **Why Spectral Gating Performs Worse Than Unprocessed Noisy Audio (Expected Limitation, Not a Bug):** On the test set, Stationary Spectral Gating scores PESQ = 1.4232 (vs. Noisy 1.9719) and STOI = 0.8234 (vs. Noisy 0.9211); on the validation set, PESQ = 1.2744 (vs. Noisy 1.5714) and STOI = 0.7392 (vs. Noisy 0.8126). This occurs for three well-known signal-processing reasons:
1. **Speech-Inflated Noise Threshold (y_noise=None):** Because VoiceBank-DEMAND utterances are short continuous sentences (mean ~2.5–2.9s) dominated by active speech rather than long noise-only pauses, computing mu_f + 1.5 * sigma_f across the full utterance includes vowel/formant energy in the noise estimate, pushing the gate threshold artificially high.
2. **Full Attenuation (prop_decrease=1.0) Clips Weak Consonants:** With prop_decrease=1.0, time-frequency bins below the inflated threshold are attenuated by 100%, erasing low-energy unvoiced consonants and formant transitions (which directly reduces STOI intelligibility).
3. **Non-Stationary DEMAND Noises & Musical Noise:** DEMAND environments (cafe, babble, living, psquare, kitchen) are non-stationary; hard stationary gating creates isolated spectral islands ("musical noise") that ITU-T P.862.2 Wideband PESQ heavily penalizes. This is a textbook limitation of classical blind stationary spectral gating and serves as the exact scientific motivation for our U-Net DNN (Demucs).

### C. Demucs dns48 Evaluation & PESQ/STOI Verification (src/evaluation/evaluator.py)

- **Inference Rigor:** Pretrained dns48 is evaluated at 16 kHz mono in model.eval() mode under torch.no_grad(), preserving exact sample length (enhanced.shape == clean.shape).
- **Metrics Rigor:** Wideband PESQ uses pesq(16000, clean_np, enh_np, "wb") and STOI uses stoi(clean_np, enh_np, 16000, extended=False). All 770 validation utterances (2,310 evaluations) and 824 test utterances (2,472 evaluations) succeeded with 0 failed samples.

### D. Test Set Integrity (p232_001.wav) & Results Reproducibility

- **p232_001.wav Audit:** Explicitly identified as the single utterance used in the Stage 2 smoke test solely to verify pipeline mechanics. Neither p232_001.wav nor any other test utterance was used for training, hyperparameter selection, or checkpoint selection. Validation metrics (results/val_baseline_summary.json, 770 utterances) are strictly locked as the sole reference for Phase 3B model selection.
- **CSV & JSON Parity:** val_baseline_per_utterance.csv contains exactly 2,311 lines (1 header + 2,310 rows = 770 x 3) and test_baseline_per_utterance.csv contains exactly 2,473 lines (1 header + 2,472 rows = 824 x 3), with 0 missing values, 0 duplicates, and 100% agreement with the summary JSON/CSV statistics.

## 4. Issues and Recommended Corrections

- **Critical Issues:** 0
- **Major Issues:** 0
- **Minor Observations (Informational Only — No Code Changes Required Before Phase 3B):**
1. **Minor Observation 1 (Spectral Gating Documentation):** When presenting the final academic comparison, explicitly note that Stationary Spectral Gating was run blind (y_noise=None, prop_decrease=1.0) without oracle noise-only reference segments, explaining why stationary thresholding attenuates weak speech formants and produces musical noise in non-stationary DEMAND environments.
2. **Minor Observation 2 (Window Count Reporting Consistency):** With segment=4.0s (64,000 samples) and stride=2.0s (32,000 samples), the 10,802 training utterances yield 12,272 sliding-window training chunks (as verified by len(train_dataset) == 12,272 in test_phase3a_preliminary.py). Ensure all training logs report the exact DataLoader length (12,272 chunks).

## 5. Final Phase 3B Readiness Verdict

### VERDICT: APPROVED — Safe to Proceed with Phase 3B (Demucs dns48 Fine-Tuning).

All dataset loaders, runtime restoration safeguards, baseline implementations, and validation/test evaluation pipelines have been independently verified with zero critical or major issues. The validation baseline benchmark (Noisy PESQ: 1.5714 / STOI: 0.8126; Pretrained dns48 Zero-Shot PESQ: 2.0691 / STOI: 0.8523 across 770 speaker-independent validation utterances) is locked and ready to govern checkpoint selection during Phase 3B fine-tuning.
