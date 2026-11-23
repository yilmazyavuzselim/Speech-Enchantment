> **Document ID:** `18` | **Phase:** Phase 5B — Offline Interactive Audio Demo | **Role:** Implementation, Packaging & Independent QA  
> **Source Word Document:** `Phase_5B_Offline_Audio_Demo_Report.docx` | **Repository Path:** `Speech-Ench-Answers/18_Phase_5B_Offline_Audio_Demo_Report.md`

---

# Phase 5B — Offline Interactive Speech Enhancement Audio Demo Report

Repository: https://github.com/yilmazyavuzselim/Speech-Enchantment (branch main). Base commit before Phase 5B: 070303390b3e8e132b109fcf319ddc5d50f63b34. Report generated: 2026-11-23T13:00:00Z.

Final Phase 5B status: COMPLETED — Independent QA verdict: APPROVED / PASS.

## 1. Objective and Scope

Provide a fully offline, self-contained HTML5 demonstration that lets listeners compare five audio conditions for four official VoiceBank-DEMAND test utterances, plus a portable ZIP package. This was a packaging task only: no training, no inference, no audio processing, no re-encoding and no PESQ/STOI recalculation were performed. Original files in results/audio_examples/ and all historical reports (01–17) were not modified.

- Workflow: Step 1 read-only inventory (Research / QA Engineer) → Step 2 implementation and packaging (Implementation Engineer) → Step 3 independent read-only QA (initial audit, then a re-audit after post-audit corrections) → Step 4 reports → Step 5 GitHub synchronization.
- Post-audit corrections: statistical values in the demo were corrected to match primary_and_secondary_tests.csv; 4 minor documentation findings (F-01 to F-04) from the re-audit were fixed; the ZIP was rebuilt and re-verified.

## 2. Demo Contents: Utterances and Conditions

| Utterance | Speaker | Noise | Input SNR | Duration | Samples | Bytes per WAV |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| p232_023 | p232 (Male) | psquare | 2.5 dB | 9.7689 s | 156,302 | 312,648 |
| p257_008 | p257 (Female) | cafe | 7.5 dB | 5.5831 s | 89,330 | 178,704 |
| p232_012 | p232 (Male) | living | 12.5 dB | 4.3357 s | 69,371 | 138,786 |
| p257_014 | p257 (Female) | office | 17.5 dB | 3.6925 s | 59,080 | 118,204 |

Five conditions per utterance (16 kHz, mono, 16-bit PCM WAV): clean.wav (reference), noisy.wav (input), spectral_gating.wav (DSP baseline), dns48_zeroshot.wav (pretrained Demucs dns48), dns48_finetuned.wav (project model, Epoch 14, best_val_pesq.th). Total: 20 WAV files, 3,741,710 bytes.

### Per-sample objective metrics shown in the demo (individual samples, PESQ / STOI)

| Utterance | Noisy | Spectral Gating | DNS48 Zero-Shot | DNS48 Fine-Tuned |
| :--- | :--- | :--- | :--- | :--- |
| p232_023 | 1.3108 / 0.9366 | 1.1323 / 0.8165 | 2.2626 / 0.9671 | 2.5008 / 0.9675 |
| p257_008 | 1.2496 / 0.9045 | 1.0906 / 0.7877 | 1.9051 / 0.9325 | 2.2039 / 0.9531 |
| p232_012 | 2.0530 / 0.9654 | 1.1952 / 0.8279 | 2.7081 / 0.9648 | 3.1346 / 0.9739 |
| p257_014 | 3.3299 / 0.9944 | 1.1785 / 0.8065 | 3.4156 / 0.9950 | 3.7012 / 0.9961 |

These are single-utterance values from results/final_test_comparison.json. They illustrate the examples only and must not be read as test-set statistics.

## 3. Offline Architecture and Usage

| Path | Bytes | Role |
| :--- | :--- | :--- |
| audio_demo/index.html | 39,950 | Single-file HTML5/CSS/vanilla JS app; inline styles and scripts |
| audio_demo/README.md | 14,414 | Usage guide, conditions, metrics |
| audio_demo/audio_integrity_manifest.json | 5,976 | SHA-256 manifest for the 20 WAVs |
| audio_demo/audio/<utterance>/<condition>.wav | 3,741,710 | 20 byte-identical copies of results/audio_examples/ |

- Audio is loaded only via native HTML5 <audio preload="metadata"> elements with relative paths audio/<utterance>/<condition>.wav. No fetch(), XMLHttpRequest or Web Audio decodeAudioData is used, so the page works from file:// without a web server.
- No external stylesheets, scripts, CDNs, web fonts or remote images. No autoplay attribute; playback starts only when the user presses Play.
- Features: utterance selector, five condition cards with play/pause/restart/seek and native controls, single-active-audio rule, A/B switcher with timestamp preservation, per-sample and aggregate metric tables, per-card error banners.
- Usage: open audio_demo/index.html in a browser, or extract the ZIP anywhere and open index.html. Keep the audio/ folder next to index.html.

## 4. Audio Integrity Verification (SHA-256)

Every hash below was read from audio_demo/audio_integrity_manifest.json and re-computed at report-generation time from both results/audio_examples/<utt>/<cond>.wav and audio_demo/audio/<utt>/<cond>.wav; all three matched for all 20 files.

| Relative path | Bytes | SHA-256 |
| :--- | :--- | :--- |
| audio/p232_023/clean.wav | 312,648 | 3be1791a7030fc4e92918447691c6084744db364842a1407fdcb603e88cb49df |
| audio/p232_023/noisy.wav | 312,648 | 81f21f5ab8422f07d7e73b75937e47b5602fb410220d30d4257dea8531ada67c |
| audio/p232_023/spectral_gating.wav | 312,648 | 9161a611784114e7842945f185731821ad58c8c6235bbbf62dfa529609b92dd5 |
| audio/p232_023/dns48_zeroshot.wav | 312,648 | aaeae81dadcae14ab56e24f05fdef768502f342f99225302eac6240764326658 |
| audio/p232_023/dns48_finetuned.wav | 312,648 | 39d8f14a8ae57fd655ddb1c3ec2c0b72803e5dfc89099b4023e2392a9af5223a |
| audio/p257_008/clean.wav | 178,704 | 2d73016d829466d6d2b840720fbafe102ec25f5a6093cc1bfe38ffef71d774b7 |
| audio/p257_008/noisy.wav | 178,704 | b05e5adc4b6f23f46a05a86a9b9d263c414c8da9cf2cddcea8b65289cb985d47 |
| audio/p257_008/spectral_gating.wav | 178,704 | 7e45d586c91654f53af878b55e5a2c6f7ba64ba0f69f1955875d87d727a9abdb |
| audio/p257_008/dns48_zeroshot.wav | 178,704 | 42fb9273d749f081b10af0ba63304a7f018384eb9c324c5fbc7b3d7f3103adab |
| audio/p257_008/dns48_finetuned.wav | 178,704 | 37756c128daaba4f5938f17bdfad5045664b8769e91f9eadeb35b122e996e285 |
| audio/p232_012/clean.wav | 138,786 | 9e3ce88b86dbdfbed548bef4f79a4159c6d1656f6e0e7b570973e14798b373c1 |
| audio/p232_012/noisy.wav | 138,786 | 9d10fd202c80641d6c1d9e5db52ef42d40b5feb3f516b1d425cca08a2bcdc911 |
| audio/p232_012/spectral_gating.wav | 138,786 | 7117325e7034d250c5b189e1fdff3615cc1def74a30c2c0441746efe7ab8e612 |
| audio/p232_012/dns48_zeroshot.wav | 138,786 | 8c50a06e1250095cd3bf114b33a189b2696e11234e8f9c6bbd61ddde1d6e8be6 |
| audio/p232_012/dns48_finetuned.wav | 138,786 | dbd4569f8d1ca1add5cc2119d7e50bed0ea76bdd5bc4ae9e6a8f59b9046ddc58 |
| audio/p257_014/clean.wav | 118,204 | 4ff11339edf8f2bb434b464d753e2b0651054a6040a2d54ece9db0973826b807 |
| audio/p257_014/noisy.wav | 118,204 | 3bf46bd5fd13448da80b83f26b465e622f43c748f6946b39639139c22d61be40 |
| audio/p257_014/spectral_gating.wav | 118,204 | 0fb199e62cf98aaf84a70da32a8492a03ad5575d990042b0d725554c4bd67b26 |
| audio/p257_014/dns48_zeroshot.wav | 118,204 | abb5af9809872663b92aebea15bc1950949dd04cf8369e7262d672451efdf3ae |
| audio/p257_014/dns48_finetuned.wav | 118,204 | 58a183e20082e5c4b6b0a40aad84a05b732f143760e458d34bfef3f4da470311 |

## 5. ZIP Creation and Extraction Tests

| Property | Value |
| :--- | :--- |
| File | Speech_Enhancement_Offline_Audio_Demo.zip (repository root) |
| Size | 3,014,851 bytes |
| SHA-256 | 30626ce83fee5c242640125750a99acc54036975bb20e26df75ce5745a08b74a |
| Entries | 23: index.html, README.md, audio_integrity_manifest.json, 20 × audio/<utt>/<cond>.wav (forward slashes only) |
| Fresh extraction test | PASS — Extracted into fresh %TEMP% directory via Expand-Archive; 20/20 extracted WAV hashes matched audio_integrity_manifest.json (3,741,710 bytes); index.html, README.md, and manifest were byte-identical to audio_demo/ sources. |
| HTML / README / manifest inside ZIP | Byte-identical (SHA-256) to audio_demo/ sources |
| Headless browser test | PASS (Headless Microsoft Edge only) — Opened extracted index.html via file:// with --headless=new; cycled all 4 utterances via selectUtterance(); 20/20 <audio> elements fired loadedmetadata with exact durations (9.7689, 5.5831, 4.3357, 3.6925 s), 0 visible error banners, 0 non-file:// resource requests, no autoplay attribute, single-active-audio rule PASS (muted programmatic play). Chrome, Firefox, Safari, and physical speaker playback: NOT VERIFIED. |
| GitHub size limit | 2.88 MiB, well under the 100 MB per-file limit; committed directly (no Release asset needed) |

## 6. Statistical Information (Aggregate Test Set, N = 824)

Source: results/statistical_analysis/primary_and_secondary_tests.csv (Phase 4D). Comparison: DNS48 Fine-Tuned (Epoch 14) vs DNS48 Zero-Shot on all 824 paired official test utterances. These statistics are not computed from the 4 demo samples. Rounding: 4 decimals (decimal half-up), p-values 3 significant figures.

| Metric | Sign Test wins | Sign Test p_Holm | Wilcoxon | Hodges–Lehmann shift [95% CI] | Mean gain [95% CI]; t-test | Effect sizes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PESQ | 748/824 (90.78%) | 2.42e-139 | Z = 22.12, p_Holm = 3.73e-108 | +0.4036 [+0.3822, +0.4248] | +0.3977 [+0.3735, +0.4214]; t(823) = 32.24, p_Holm = 8.55e-148 | r_rb = +0.8897; d_z = 1.1231 |
| STOI | 669/824 (81.19%) | 3.72e-81 | Z = 19.01, p_Holm = 1.37e-80 | +0.0136 [+0.0117, +0.0155] | +0.0179 [+0.0161, +0.0198]; t(823) = 18.54, p_Holm = 2.24e-64 | r_rb = +0.7692; d_z = 0.6458 |

- Statistical significance (very small p-values with N = 824) is distinct from perceptual or practical improvement; Phase 4D treats practical thresholds as exploratory. Inference is scoped to this benchmark (two test speakers, p232 and p257).

## 7. Independent QA Results

| Area | Result |
| :--- | :--- |
| Area 1: Statistical Accuracy | PASS — 30/30 displayed values verified against results/statistical_analysis/primary_and_secondary_tests.csv (PESQ & STOI Sign Test, Wilcoxon, Hodges-Lehmann shift & 95% CI, Paired t-test, r_rb, Cohen's d_z); aggregate N=824 scope clearly distinguished from 4 demo samples. |
| Area 2: HTML vs. README Consistency | PASS — Exact numerical and textual parity across audio_demo/index.html and audio_demo/README.md. |
| Area 3: ZIP Contents & Structure | PASS — 23 entries (index.html, README.md, audio_integrity_manifest.json, 20 WAVs), 3,014,851 bytes, forward slashes only, zero datasets/checkpoints/logs/scripts. |
| Area 4: SHA-256 Integrity | PASS — 20/20 WAV files in results/audio_examples/, audio_demo/audio/, audio_integrity_manifest.json, and extracted ZIP match byte-for-byte (3,741,710 bytes). |
| Area 5: Audio-to-Utterance Mapping | PASS — 4 utterances (p232_023, p257_008, p232_012, p257_014) x 5 conditions (clean, noisy, spectral_gating, dns48_zeroshot, dns48_finetuned), 16 kHz 16-bit PCM mono. |
| Area 6: Relative Paths & file:// Safety | PASS — Strictly relative audio/<utt>/<cond>.wav paths; zero absolute, localhost, or remote URLs. |
| Area 7: Extraction & Headless Browser Evidence | PASS — Fresh %TEMP% extraction and headless Microsoft Edge offline load verified (20/20 loadedmetadata, single-active-audio PASS); Chrome/Firefox/Safari and live acoustic playback explicitly marked NOT VERIFIED. |
| Area 8: Zero External Dependencies | PASS — Inline CSS/JS, zero CDNs/fonts/images, zero fetch()/XHR/decodeAudioData calls, no autoplay. |
| Area 9: Source Preservation & Hygiene (incl. F-01 to F-04) | PASS — All protected assets, results/, scripts/01-07, and historical reports 01-17 untouched; all 4 minor documentation findings (F-01 to F-04) fixed and re-verified. |

Re-audit findings F-01 to F-04 (all MINOR, documentation only) were fixed by the Implementation Engineer and confirmed in a targeted re-check. Final verdict: APPROVED / PASS.

## 8. Known Limitations and Unverified Items

- VERIFIED: 20/20 WAV SHA-256 parity (source, demo copy, manifest, extracted ZIP); ZIP structure; statistics vs CSV; offline loading of all 20 files in headless Microsoft Edge via file://; single-active-audio rule (muted programmatic play).
- NOT VERIFIED: playback in Google Chrome, Mozilla Firefox and Apple Safari; real acoustic output through speakers/headphones and subjective listening quality; interactive mouse/keyboard UI use by a human.
- The ZIP stores files at its root (no top-level folder); on some unzip tools files extract into the current directory.
- Per-sample values describe four hand-picked examples and are not representative estimates.