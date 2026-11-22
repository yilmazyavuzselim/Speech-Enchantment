> **Document ID:** `13` | **Phase:** Phase 4 — Official Test-Set Evaluation | **Role:** Implementation Engineer Report  
> **Source Word Document:** `Phase_4_Official_Test_Set_Evaluation_Final_Report.docx` | **Repository Path:** `Speech-Ench-Answers/13_Phase_4_Official_Test_Set_Evaluation_Final_Report.md`

---
# Phase 4 Final Report: Official Test-Set Evaluation (Speech-Enhancement)

**Executed By:** Implementation Engineer Subagent (impl-engineer, Model: Pro, Workspace: inherit, Write enabled, Colab NVIDIA L4 GPU)

## 1. Exact Checkpoint Used

- **Primary Selected Checkpoint:** `/content/drive/MyDrive/Speech-Enhancement/checkpoints/best_val_pesq.th`
- **Google Drive ID:** 1e0VGfFb4exie8aqBTyvV9Gq0Ey3W4jlj (226,478,148 bytes / 216.0 MiB).
- **Selection Criterion:** Selected strictly on the 770-utterance validation split (p226, p287) at Epoch 14 (where validation Wideband PESQ reached its peak of 2.5691 ± 0.6668).

## 2. Pre-Evaluation Checkpoint Verification

Verified automatically during the Phase 4 preflight check (job_024_phase4_preflight) prior to running the full test evaluation:

- **Checkpoint Epoch & Tensors:** epoch == 14, 48 parameter tensors loaded with strict=True into Demucs (18,867,937 parameters).
- **Architecture Attributes:** `chin=1, chout=1, hidden=48, depth=5, kernel_size=8, stride=4, causal=True, resample=4, normalize=True, sample_rate=16000.`
- **Execution Mode:** model.eval() (model.training == False) under torch.no_grad() on cuda (NVIDIA L4).

## 3. Pre-Evaluation Test Dataset & Historical Baseline Verification

- **Test Manifest (manifests/test_manifest.json):** Verified exactly 824 clean/noisy utterance pairs from the 2 unseen test speakers (p232: 393 utterances; p257: 431 utterances; total duration: 2,072.1s / 0.576 hours). All 824 clean and noisy audio paths resolved in /content/data_16k/, all verified as 16,000 Hz, 16-bit PCM mono, with 100% sample-count alignment and 0 missing or corrupted files.
- **Historical Phase 3A Baseline Preservation (results/test_baseline_per_utterance.csv):** Verified untouched (2,472 data rows = 824 utterances x 3 conditions: noisy, spectral_gating, dns48_zeroshot), with exact 1-to-1 utterance_id parity against test_manifest.json. Phase 3A baseline results were reused directly without recomputation or modification.
- **Explicit Disclosure Regarding p232_001.wav:** Test utterance p232_001.wav ("Please call Stella.", bus noise at 17.5 dB SNR, 27,861 samples) was previously used in the Stage 2 single-pair smoke test solely to verify pipeline mechanics. It was never used for training, hyperparameter tuning, or checkpoint selection, and was retained and evaluated normally (Noisy PESQ: 2.9238 / STOI: 0.8967 -> Spectral Gating PESQ: 1.6644 / STOI: 0.8554 -> Zero-Shot PESQ: 3.0293 / STOI: 0.9017 -> Fine-Tuned PESQ: 3.1112 / STOI: 0.9085).

## 4. Evaluation Settings & 5-Sample Preflight Gate

- **Preprocessing & Inference:** Identical to Phase 3A—full uncropped 16 kHz mono noisy waveforms passed to model(noisy) on cuda (NVIDIA L4) under model.eval() + torch.no_grad(), verifying est.shape == clean.shape and torch.isfinite(est).all().
- **Metrics:** Wideband PESQ: pesq(16000, clean_np, enh_np, "wb") and STOI: stoi(clean_np, enh_np, 16000, extended=False).
- **5-Sample Preflight Check (job_024_phase4_preflight):** Passed on p257_001 through p257_006 (10.37s, all finite, 0 errors) before launching the full 824-utterance evaluation (job_025_phase4_full_eval).

## 5. Number of Successfully Evaluated Utterances

- **Fine-Tuned Model (dns48_finetuned, Phase 4):** 824 / 824 successful (0 failed).
- **Unified 4-Way Comparison Total:** 3,296 / 3,296 evaluations (824 utterances x 4 methods, 0 failed).

## 6. Overall 4-Way Comparison Table (results/final_test_comparison.csv)

| Method / Condition | Provenance | PESQ Mean ± Std | PESQ Med | PESQ Min – Max | STOI Mean ± Std | STOI Med | STOI Min – Max | Success |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1. Noisy Audio (Unprocessed) | Phase 3A Baseline | 1.9719 ± 0.7532 | 1.7986 | 1.0313 – 4.3025 | 0.9211 ± 0.0709 | 0.9403 | 0.5587 – 0.9998 | 824 / 0 |
| 2. Stationary Spectral Gating | Phase 3A Baseline | 1.4232 ± 0.2848 | 1.3502 | 1.0391 – 2.6532 | 0.8234 ± 0.0655 | 0.8320 | 0.4614 – 0.9630 | 824 / 0 |
| 3. Pretrained Demucs dns48 (Zero-Shot) | Phase 3A Baseline | 2.5203 ± 0.6258 | 2.5263 | 1.1491 – 3.9552 | 0.9300 ± 0.0638 | 0.9476 | 0.5605 – 0.9997 | 824 / 0 |
| 4. Fine-Tuned Demucs dns48 (Epoch 14) | Phase 4 (New) | 2.9180 ± 0.6483 | 3.0316 | 1.1415 – 4.1229 | 0.9480 ± 0.0507 | 0.9645 | 0.6812 – 0.9998 | 824 / 0 |

## 7. Wideband PESQ Improvements (dns48_finetuned, N = 824)

| Comparison Pair | Absolute Delta PESQ (Mean) | Absolute Delta PESQ (Median) | Relative Mean Change (%) | Utterance Win Rate (PESQ_FT > PESQ_Ref) |
| :--- | :--- | :--- | :--- | :--- |
| Fine-Tuned dns48 vs. Noisy Audio | +0.9461 (2.9180 vs. 1.9719) | +1.2330 (3.0316 vs. 1.7986) | +47.98% | 810 / 824 (98.30%) |
| Fine-Tuned dns48 vs. Pretrained dns48 Zero-Shot | +0.3977 (2.9180 vs. 2.5203) | +0.5053 (3.0316 vs. 2.5263) | +15.78% | 748 / 824 (90.78%) |
| Fine-Tuned dns48 vs. Stationary Spectral Gating | +1.4948 (2.9180 vs. 1.4232) | +1.6814 (3.0316 vs. 1.3502) | +105.03% | 824 / 824 (100.00%) |

## 8. STOI Intelligibility Improvements (dns48_finetuned, N = 824)

| Comparison Pair | Absolute Delta STOI (Mean) | Absolute Delta STOI (Median) | Relative Mean Change (%) |
| :--- | :--- | :--- | :--- |
| Fine-Tuned dns48 vs. Noisy Audio | +0.0269 (0.9480 vs. 0.9211) | +0.0242 (0.9645 vs. 0.9403) | +2.92% |
| Fine-Tuned dns48 vs. Pretrained dns48 Zero-Shot | +0.0180 (0.9480 vs. 0.9300) | +0.0169 (0.9645 vs. 0.9476) | +1.94% |
| Fine-Tuned dns48 vs. Stationary Spectral Gating | +0.1246 (0.9480 vs. 0.8234) | +0.1325 (0.9645 vs. 0.8320) | +15.13% |

## 9. Speaker-Level Breakdown (results/test_breakdown_by_speaker.csv)

| Test Speaker | Utts | Noisy PESQ / STOI | Spectral Gating PESQ / STOI | dns48 Zero-Shot PESQ / STOI | dns48 Fine-Tuned PESQ / STOI | Delta PESQ (FT vs Noisy / ZS) | Delta STOI (FT vs Noisy / ZS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| p232 (Male) | 393 | 2.1687 / 0.9270 | 1.3991 / 0.8334 | 2.6535 / 0.9400 | 3.0351 ± 0.5971 / 0.9504 ± 0.0504 | +0.8664 / +0.3817 | +0.0235 / +0.0104 |
| p257 (Female) | 431 | 1.7925 / 0.9157 | 1.4451 / 0.8143 | 2.3989 / 0.9209 | 2.8112 ± 0.6742 / 0.9457 ± 0.0508 | +1.0187 / +0.4123 | +0.0300 / +0.0248 |

## 10. Noise-Level Breakdown (results/test_breakdown_by_noise.csv)

| Unseen Noise | Utts | Noisy PESQ / STOI | Spectral Gating PESQ / STOI | dns48 Zero-Shot PESQ / STOI | dns48 Fine-Tuned PESQ / STOI | Delta PESQ (FT vs Noisy / ZS) | Delta STOI (FT vs Noisy / ZS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| office | 164 | 2.5361 / 0.9583 | 1.5682 / 0.8432 | 2.9006 / 0.9498 | 3.4342 ± 0.4769 / 0.9667 ± 0.0464 | +0.8982 / +0.5337 | +0.0084 / +0.0169 |
| bus | 164 | 2.4845 / 0.9519 | 1.5220 / 0.8401 | 2.8184 / 0.9488 | 3.2107 ± 0.4532 / 0.9635 ± 0.0405 | +0.7262 / +0.3922 | +0.0116 / +0.0146 |
| psquare | 166 | 1.7367 / 0.9145 | 1.3738 / 0.8211 | 2.4527 / 0.9266 | 2.8194 ± 0.5205 / 0.9430 ± 0.0490 | +1.0827 / +0.3666 | +0.0285 / +0.0165 |
| living | 166 | 1.6178 / 0.9040 | 1.3628 / 0.8202 | 2.3150 / 0.9218 | 2.6632 ± 0.6283 / 0.9411 ± 0.0455 | +1.0454 / +0.3482 | +0.0371 / +0.0193 |
| cafe | 164 | 1.4919 / 0.8768 | 1.2904 / 0.7925 | 2.1181 / 0.9032 | 2.4668 ± 0.6115 / 0.9256 ± 0.0586 | +0.9749 / +0.3487 | +0.0488 / +0.0224 |

## 11. SNR-Level Breakdown (results/test_breakdown_by_snr.csv)

| Test SNR | Utts | Noisy PESQ / STOI | Spectral Gating PESQ / STOI | dns48 Zero-Shot PESQ / STOI | dns48 Fine-Tuned PESQ / STOI | Delta PESQ (FT vs Noisy / ZS) | Delta STOI (FT vs Noisy / ZS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2.5 dB | 207 | 1.4232 / 0.8683 | 1.2810 / 0.7860 | 1.9860 / 0.8907 | 2.4203 ± 0.6152 / 0.9190 ± 0.0604 | +0.9971 / +0.4343 | +0.0507 / +0.0283 |
| 7.5 dB | 206 | 1.7651 / 0.9188 | 1.3968 / 0.8267 | 2.3774 / 0.9260 | 2.8218 ± 0.5718 / 0.9474 ± 0.0444 | +1.0567 / +0.4444 | +0.0287 / +0.0214 |
| 12.5 dB | 207 | 2.1041 / 0.9403 | 1.4446 / 0.8351 | 2.6599 / 0.9444 | 3.0631 ± 0.5385 / 0.9582 ± 0.0398 | +0.9590 / +0.4032 | +0.0179 / +0.0138 |
| 17.5 dB | 204 | 2.6035 / 0.9574 | 1.5722 / 0.8462 | 3.0651 / 0.9594 | 3.3728 ± 0.4476 / 0.9674 ± 0.0416 | +0.7693 / +0.3078 | +0.0101 / +0.0081 |

## 12. Utterance-Level Diagnostic & Failure-Case Analysis

- **Fine-Tuning vs. Zero-Shot (dns48):** Fine-tuning improved Wideband PESQ on 748 / 824 utterances (90.78%) and did not improve over zero-shot on 76 / 824 utterances (9.22%).
- **Fine-Tuning vs. Unprocessed Noisy:** Fine-tuning improved Wideband PESQ on 810 / 824 utterances (98.30%) and had lower PESQ than unprocessed noisy on only 14 / 824 utterances (1.70%).
- **PESQ Improvement with Minor STOI Trade-off:** Occurred on 122 / 824 utterances (14.81%) vs. Zero-Shot and 78 / 824 utterances (9.47%) vs. Noisy (where high-SNR STOI was already near ceiling).
- **Top PESQ Gains over Zero-Shot:** p257_083 (bus, 17.5 dB: +1.9962 PESQ, 1.7045 -> 3.7007) and p257_158 (office, 7.5 dB: +1.6610 PESQ, 1.9688 -> 3.6298).

## 13. Comparison With Validation Results

| Split (Speakers) | Utts | SNR Range (dB) | Noisy PESQ / STOI | Zero-Shot dns48 PESQ / STOI | Fine-Tuned dns48 (Ep 14) PESQ / STOI | Delta PESQ (FT vs Noisy / ZS) | Delta STOI (FT vs Noisy / ZS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Validation Set (p226, p287) | 770 | 0, 5, 10, 15 | 1.5714 / 0.8126 | 2.0691 / 0.8523 | 2.5691 / 0.8835 | +0.9977 / +0.5000 | +0.0709 / +0.0312 |
| Official Test Set (p232, p257) | 824 | 2.5, 7.5, 12.5, 17.5 | 1.9719 / 0.9211 | 2.5203 / 0.9300 | 2.9180 / 0.9480 | +0.9461 / +0.3977 | +0.0269 / +0.0180 |

## 14. Limitations & Methodological Caveats

1. **Descriptive Statistics Only:** Per project rules, all reported metrics and deltas are exact empirical sample statistics across the 824 test utterances without claiming formal hypothesis-test significance levels.
2. **Disclosure of p232_001.wav:** Utterance p232_001.wav was inspected once during the Stage 2 real-audio smoke test for pipeline verification; it was never used for training or model selection and is explicitly reported in Section 3.

## 15. Files Created and Their Locations

- `scripts/07_evaluate_and_report.py`
- `results/test_finetuned_per_utterance.csv (824 rows)`
- `results/test_finetuned_summary.json`
- `results/final_test_comparison.csv & results/final_test_comparison.json`
- `results/test_breakdown_by_speaker.csv, results/test_breakdown_by_noise.csv, results/test_breakdown_by_snr.csv`
- `results/final_test_comparison.png`
- `results/audio_examples/ (p232_023, p257_008, p232_012, p257_014 — 4 representative utterances x 5 conditions = 20 WAV files)`
- `README.md`
