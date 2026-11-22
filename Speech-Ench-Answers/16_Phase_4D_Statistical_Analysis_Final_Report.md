> **Document ID:** `16` | **Phase:** Phase 4D — Formal Statistical Hypothesis Testing & Effect-Size Report | **Role:** Statistical Analysis & QA Verification  
> **Source Word Document:** `Phase_4D_Statistical_Analysis_Final_Report.docx` | **Repository Path:** `Speech-Ench-Answers/16_Phase_4D_Statistical_Analysis_Final_Report.md`

---
# Phase 4D Formal Statistical Hypothesis Testing, Symmetry Diagnostics & Practical Effect-Size Report

**Verified By:** Research / QA Engineer Subagent (qa-research-engineer, Model: Pro, Workspace: inherit, Write access: DISABLED, Strictly read-only)

## 1. Executive Summary & Methodological Compliance

- **QA Verification Verdict: APPROVED / PASS (100% Mathematical Consistency & Zero Modification of Historical Artifacts)**
- **Primary Comparison:** Demucs dns48 Fine-Tuned (Epoch 14, best_val_pesq.th) vs. Demucs dns48 Zero-Shot across all N = 824 paired official test utterances (d_i = x_{i,FT} - x_{i,ZS}).
- **All 4 Methodological Corrections Incorporated:**
1. **Test Hierarchy & Symmetry Diagnostics:** The exact two-sided Paired Sign Test is treated as the primary symmetry-free test of median difference (H_0: Median(d_i) = 0). The Wilcoxon Signed-Rank test is executed as a complementary location-shift test for the Hodges–Lehmann pseudo-median (theta_HL) only after evaluating symmetry of d_i via histograms, Q-Q plots, moment skewness (g_1), Bowley quartile skewness, Kelly decile skewness, symmetry identity R^2, and |theta_HL - Median|/IQR. We explicitly show why Shapiro–Wilk / D'Agostino K^2 normality tests alone do not establish Wilcoxon suitability. The Paired Student's t-test is reported as a secondary parametric mean-difference reference.
2. **Speaker Clustering & Scope of Inference:** Speaker-stratified bootstrap (B = 10,000 resamples, seed = 42) and separate analyses for speaker p232 (n = 393) and speaker p257 (n = 431) are reported. Because resampling utterances within only 2 fixed speaker clusters cannot estimate between-speaker variance across a general human population, pooled inference is strictly scoped to this benchmark's 824 test utterances and 2 held-out speakers.
3. **Exploratory Operational Thresholds & Literature Citations:** Practical cutoffs are explicitly labeled as Exploratory Operational Thresholds (informed by subjective MOS residual uncertainty ~0.10–0.25 MOS-LQO in ITU-T Rec. P.862.2 [2007], ITU-T Rec. P.800.1, Hu & Loizou [2008], Loizou [2013], and logistic intelligibility saturation in Taal et al. [2011]) rather than universally mandated JND/MCID constants, accompanied by a full multi-threshold sensitivity grid.
4. **Predefined Non-Overlapping Category Boundaries:** Five mutually exclusive, collectively exhaustive categories (Substantial Improvement, Moderate Improvement, Operational Equivalence / ROPE, Moderate Degradation, Substantial Degradation) were predefined before counting and sum to exactly N = 824 (100.00%), including pre-specified Low-SNR (2.5 & 7.5 dB, n = 413) and High-SNR Ceiling (12.5 & 17.5 dB, n = 411) STOI regimes.
- **Separate Output Directory & Artifact Preservation:** All new statistical artifacts were written exclusively to results/statistical_analysis/. Before-and-after SHA-256 verification confirms that all 18 historical Phase 3A, 3B, 3C, 4, and 4C files remain 100% unmodified.

## 2. Pre-Test Symmetry & Distributional Diagnostics of Paired Differences (d_i)

**Why Shapiro–Wilk and D'Agostino K^2 Alone Do Not Establish Wilcoxon Suitability:** Normality tests evaluate H_0: d_i ~ N(mu, sigma^2) and reject H_0 whenever a distribution exhibits either heavy tails (leptokurtosis, g_2 != 0) or asymmetry (skewness, g_1 != 0). However, the Wilcoxon Signed-Rank test does NOT require normality—it requires symmetry of d_i around its central location parameter theta. A distribution that is heavy-tailed but symmetric around its median will strongly reject Shapiro–Wilk / D'Agostino normality while remaining ideal for the Wilcoxon Signed-Rank test. Conversely, a distribution with pronounced skewness violates the Wilcoxon assumption of symmetry around the median, causing the Hodges–Lehmann pseudo-median (theta_HL) to diverge from the true sample median Median(d_i).

| Comparison (N = 824) | Mean / Median / HL (theta_HL) | \|HL - Med\| (% of IQR) | Moment Skew (g_1) / Kurt (g_2) | Bowley / Kelly Skew | Symmetry R^2 / Q-Q r | D'Agostino K^2 (p-val) | Symmetry Assessment for Wilcoxon |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Primary: PESQ (FT vs. ZS) | +0.3977 / +0.3977 / +0.4036 | 0.0059 (1.52%) | -0.4461 / +2.9210 | +0.0298 / +0.0345 | 0.9627 / 0.9799 | 87.0845 (1.23e-19) | Symmetric around median (Bowley +0.03); non-normality is heavy-tailed kurtosis. Wilcoxon well-suited. |
| Primary: STOI (FT vs. ZS) | +0.01795 / +0.00845 / +0.01360 | 0.00515 (17.94%) | +1.4740 / +3.2091 | +0.4599 / +0.6276 | 0.5095 / 0.9251 | 239.5876 (9.42e-53) | Right-skewed due to ceiling compression near 1.0. Prioritize Paired Sign Test for median; Wilcoxon tests pseudo-median. |
| Secondary: PESQ (FT vs. Noisy) | +0.9460 / +0.9545 / +0.9465 | 0.0080 (1.29%) | +0.0105 / -0.1313 | -0.0432 / -0.0275 | 0.9963 / 0.9996 | 0.5157 (0.772696) | Near-perfect Gaussian & symmetric distribution (p = 0.77). All 3 tests agree identically. |
| Secondary: STOI (FT vs. Noisy) | +0.02690 / +0.01630 / +0.02215 | 0.00585 (15.84%) | +1.3300 / +2.0691 | +0.3433 / +0.5406 | 0.5019 / 0.9352 | 193.4350 (9.91e-43) | Right-skewed due to high-SNR ceiling compression. Prioritize Sign Test for median; Wilcoxon tests pseudo-median. |
| Secondary: PESQ (FT vs. SG) | +1.4948 / +1.5446 / +1.5104 | 0.0342 (4.15%) | -0.2509 / -0.3391 | -0.1010 / -0.1175 | 0.9141 / 0.9964 | 13.8260 (9.95e-4) | Mild asymmetry (\|g1\|=0.25, \|Bowley\|=0.10); Sign Test and Wilcoxon both valid. |
| Secondary: STOI (FT vs. SG) | +0.1245 / +0.1237 / +0.1235 | 0.0002 (0.37%) | +0.3168 / +1.2868 | -0.0169 / +0.0028 | 0.9823 / 0.9926 | 37.9734 (5.68e-9) | Highly symmetric around median (Bowley -0.017, HL - Med = 0.0002). Wilcoxon well-suited. |

## 3. Primary (Family 1, m = 2) & Secondary (Family 2, m = 4) Hypothesis Testing Results

**Exact Pairing, Zero Differences, and Tied Ranks:** Across all N = 824 paired test utterances (393 from p232, 431 from p257), zero missing values occurred. Due to 4-decimal CSV precision:

- **Primary Wideband PESQ (FT vs. ZS):** N = 824, N^+ = 748 (90.78%), N^- = 76 (9.22%), N^0 = 0 exact zeros (N_nonzero = 824). Tied absolute difference groups: 30 groups involving 61 observations (tie-correction sum sum(t^3 - t) = 198).
- **Primary STOI (FT vs. ZS):** N = 824, N^+ = 669 (81.19%), N^- = 145 (17.60%), N^0 = 10 exact zeros (1.21%, N_nonzero = 814). Tied absolute difference groups: 155 groups involving 572 observations (tie-correction sum sum(t^3 - t) = 25,266). Sensitivity check between Wilcoxon zero-exclusion (Z = 19.0113, p = 1.3748e-80) and Pratt zero-ranking (Z_Pratt = 19.0498, p_Pratt = 6.5880e-81) confirms complete invariance to zero handling.

| Family & Comparison (N = 824) | Counts: N+ / N- / N0 (Tied Obs) | Primary Test: Paired Sign Test (p_raw / p_Holm) | Complementary: Wilcoxon W+ / W- / Z (p_raw / p_Holm) | Secondary Ref: Paired t-Test t (df=823) (p_raw / p_Holm) | Holm FWER Decision (alpha=0.05) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Family 1 (m=2): PESQ (FT vs. ZS) | 748 / 76 / 0 (61 tied) | k=76; p=1.2121e-139 / p_Holm=2.4242e-139 | W+=321,151.5, W-=18,748.5, Z=22.1238 (1.8660e-108 / 3.7321e-108) | t=32.2383 (4.2752e-148 / 8.5504e-148) | REJECT H0 (Significant) |
| Family 1 (m=2): STOI (FT vs. ZS) | 669 / 145 / 10 (572 tied) | k=145; p=3.7209e-81 / p_Holm=3.7209e-81 | W+=293,425.0, W-=38,280.0, Z=19.0113 (1.3748e-80 / 1.3748e-80) | t=18.5376 (2.2416e-64 / 2.2416e-64) | REJECT H0 (Significant) |
| Family 2 (m=4): PESQ (FT vs. Noisy) | 810 / 14 / 0 (36 tied) | k=14; p=1.2423e-218 / p_Holm=2.4845e-218 | W+=339,421.0, W-=479.0, Z=24.7970 (9.6627e-136 / 1.9325e-135) | t=58.8282 (4.8016e-297 / 9.6032e-297) | REJECT H0 (Significant) |
| Family 2 (m=4): STOI (FT vs. Noisy) | 730 / 84 / 10 (538 tied) | k=84; p=2.2942e-129 / p_Holm=2.2942e-129 | W+=314,783.0, W-=16,922.0, Z=22.1941 (3.9185e-109 / 3.9185e-109) | t=23.4995 (7.7178e-94 / 7.7178e-94) | REJECT H0 (Significant) |
| Family 2 (m=4): PESQ (FT vs. SG) | 818 / 6 / 0 (23 tied) | k=6; p=7.6879e-234 / p_Holm=2.3064e-233 | W+=339,854.0, W-=46.0, Z=24.8603 (1.9990e-136 / 7.9962e-136) | t=71.0329 (2.5673e-353 / 7.7020e-353) | REJECT H0 (Significant) |
| Family 2 (m=4): STOI (FT vs. SG) | 819 / 5 / 0 (335 tied) | k=5; p=5.6252e-236 / p_Holm=2.2501e-235 | W+=339,844.0, W-=56.0, Z=24.8589 (2.0727e-136 / 7.9962e-136) | t=75.7810 (2.2251e-373 / 8.9004e-373) | REJECT H0 (Significant) |

## 4. Effect Sizes & Speaker-Stratified Bootstrap 95% Confidence Intervals

**Bootstrap Methodology & Clustering Scope:** Speaker-stratified bootstrap confidence intervals were computed over B = 10,000 iterations (deterministic Mulberry32 PRNG, seed = 42), resampling n_232 = 393 utterances with replacement from speaker p232 and n_257 = 431 utterances with replacement from speaker p257 in each iteration. Unstratified bootstrap and analytic intervals are also shown below. As required by Methodological Correction 2, resampling within only 2 fixed speaker clusters accounts for within-speaker utterance variation but does not estimate between-speaker population variance; therefore, pooled intervals are strictly scoped to the 824 utterances and 2 speakers of the official Edinburgh VoiceBank-DEMAND test benchmark.

| Comparison | Median Shift (IQR) & Stratified Boot 95% CI | Hodges-Lehmann Shift & Wilcoxon 95% CI | Mean Shift ± SD & Stratified Boot 95% CI (Analytic t 95% CI) | Win Rate (Wilson 95% CI) & CLES | Rank-Biserial r_rb / Std r | Cohen's d_z (Hedges' g_z) & Boot 95% CI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Primary: PESQ (FT vs. ZS) | +0.3977 (0.3871) [+0.3736, +0.4259] | +0.4036 [+0.3822, +0.4248] | +0.3977 ± 0.3541 [+0.3735, +0.4214] ([+0.3735, +0.4219]) | 90.78% [88.61%, 92.57%] (CLES 0.9078) | 0.8897 / 0.7707 | 1.1231 (1.1221) [1.0116, 1.2470] |
| Primary: STOI (FT vs. ZS) | +0.00845 (0.0287) [+0.0066, +0.0101] | +0.01360 [+0.0117, +0.0155] | +0.01795 ± 0.02779 [+0.0161, +0.0198] ([+0.0160, +0.0198]) | 81.19% [78.38%, 83.71%] (CLES 0.8180) | 0.7692 / 0.6623 | 0.6458 (0.6452) [0.5877, 0.7082] |
| Secondary: PESQ (FT vs. Noisy) | +0.9545 (0.6178) [+0.9157, +0.9865] | +0.9465 [+0.9142, +0.9789] | +0.9460 ± 0.4616 [+0.9155, +0.9768] ([+0.9145, +0.9776]) | 98.30% [97.17%, 98.98%] (CLES 0.9830) | 0.9972 / 0.8638 | 2.0494 (2.0475) [1.9261, 2.1818] |
| Secondary: STOI (FT vs. Noisy) | +0.01630 (0.0369) [+0.0144, +0.0192] | +0.02215 [+0.0200, +0.0244] | +0.02690 ± 0.03286 [+0.0248, +0.0292] ([+0.0247, +0.0291]) | 88.59% [86.24%, 90.59%] (CLES 0.8920) | 0.8980 / 0.7732 | 0.8186 (0.8179) [0.7429, 0.9001] |
| Secondary: PESQ (FT vs. SG) | +1.5446 (0.8251) [+1.4827, +1.5905] | +1.5104 [+1.4670, +1.5538] | +1.4948 ± 0.6041 [+1.4548, +1.5350] ([+1.4535, +1.5361]) | 99.27% [98.42%, 99.66%] (CLES 0.9927) | 0.9997 / 0.8661 | 2.4746 (2.4723) [2.3533, 2.6074] |
| Secondary: STOI (FT vs. SG) | +0.12370 (0.0547) [+0.1198, +0.1262] | +0.12350 [+0.1205, +0.1265] | +0.12453 ± 0.04717 [+0.1213, +0.1277] ([+0.1213, +0.1278]) | 99.39% [98.59%, 99.74%] (CLES 0.9939) | 0.9997 / 0.8660 | 2.6400 (2.6375) [2.4674, 2.8300] |

## 5. Exploratory Subgroup Analysis (Family 3, m = 22: Holm FWER & Benjamini-Hochberg FDR)

**Labeling Note (Exploratory Analysis):** All 22 subgroup comparisons below (2 speakers, 5 unseen noise environments, 4 unseen SNR levels x 2 metrics) are explicitly labeled as Exploratory Subgroup Analyses and adjusted across all m = 22 tests simultaneously using both the Holm–Bonferroni step-down procedure (FWER alpha = 0.05) and the Benjamini–Hochberg step-up procedure (FDR q = 0.05). All 22 exploratory tests reject H_0 after Holm–Bonferroni correction (largest adjusted p_Holm = 1.2072e-9).

| Exploratory Subgroup (FT vs. ZS) | Metric (N) | N+ / N- / N0 | Mean / Median / HL Shift | Sign Test p_raw / p_Holm / q_BH | Wilcoxon Z (p_raw / p_Holm) | Win Rate / r_rb / d_z |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Speaker p232 (Male) | PESQ (393) | 370 / 23 / 0 | +0.3817 / +0.3746 / +0.3843 | 9.94e-82 / 2.19e-80 / 2.19e-80 | 16.3819 (2.57e-60 / 5.66e-59) | 94.15% / 0.9536 / 1.4778 |
| Speaker p232 (Male) | STOI (393) | 291 / 94 / 8 | +0.0104 / +0.0032 / +0.0065 | 1.52e-24 / 1.37e-23 / 2.39e-24 | 10.8658 (1.68e-27 / 2.85e-26) | 74.05% / 0.6390 / 0.4843 |
| Speaker p257 (Female) | PESQ (431) | 378 / 53 / 0 | +0.4123 / +0.4195 / +0.4245 | 1.48e-61 / 2.96e-60 / 1.08e-60 | 15.0400 (4.02e-51 / 8.03e-50) | 87.70% / 0.8361 / 0.9752 |
| Speaker p257 (Female) | STOI (431) | 378 / 51 / 2 | +0.0248 / +0.0160 / +0.0209 | 8.75e-63 / 1.84e-61 / 9.62e-62 | 15.3126 (6.30e-53 / 1.32e-51) | 87.70% / 0.8532 / 0.8019 |
| Unseen Noise: bus | PESQ (164) | 149 / 15 / 0 | +0.3922 / +0.3585 / +0.3842 | 6.27e-29 / 7.52e-28 / 1.25e-28 | 9.9359 (2.91e-23 / 2.91e-22) | 90.85% / 0.8946 / 1.0467 |
| Unseen Noise: bus | STOI (164) | 127 / 35 / 2 | +0.0146 / +0.0045 / +0.0089 | 1.85e-13 / 5.55e-13 / 2.04e-13 | 7.7211 (1.17e-14 / 3.50e-14) | 77.44% / 0.6995 / 0.5571 |
| Unseen Noise: cafe | PESQ (164) | 145 / 19 / 0 | +0.3487 / +0.3579 / +0.3610 | 3.30e-25 / 3.30e-24 / 5.58e-25 | 9.6272 (6.14e-22 / 5.52e-21) | 88.41% / 0.8668 / 1.0380 |
| Unseen Noise: cafe | STOI (164) | 138 / 26 / 0 | +0.0224 / +0.0141 / +0.0186 | 1.24e-19 / 4.95e-19 / 1.43e-19 | 9.1519 (5.59e-20 / 3.91e-19) | 84.15% / 0.8240 / 0.7442 |
| Unseen Noise: living | PESQ (166) | 144 / 22 / 0 | +0.3482 / +0.3910 / +0.3791 | 3.63e-23 / 2.91e-22 / 5.33e-23 | 9.1270 (7.05e-20 / 4.23e-19) | 86.75% / 0.8168 / 0.9934 |
| Unseen Noise: living | STOI (166) | 139 / 25 / 2 | +0.0193 / +0.0114 / +0.0164 | 2.29e-20 / 1.14e-19 / 2.80e-20 | 8.8408 (9.50e-19 / 3.80e-18) | 83.73% / 0.7960 / 0.7030 |
| Unseen Noise: office | PESQ (164) | 157 / 7 / 0 | +0.5337 / +0.5346 / +0.5303 | 4.97e-38 / 8.95e-37 / 2.19e-37 | 10.4941 (9.19e-26 / 1.19e-24) | 95.73% / 0.9449 / 1.4239 |
| Unseen Noise: office | STOI (164) | 139 / 21 / 4 | +0.0169 / +0.0048 / +0.0107 | 1.54e-22 / 1.08e-21 / 2.12e-22 | 9.0432 (1.52e-19 / 7.61e-19) | 84.76% / 0.8243 / 0.6069 |
| Unseen Noise: psquare | PESQ (166) | 153 / 13 / 0 | +0.3666 / +0.3580 / +0.3752 | 1.68e-31 / 2.19e-30 / 3.70e-31 | 10.1379 (3.75e-24 / 4.12e-23) | 92.17% / 0.9073 / 1.2264 |
| Unseen Noise: psquare | STOI (166) | 126 / 38 / 2 | +0.0165 / +0.0063 / +0.0124 | 3.21e-12 / 6.43e-12 / 3.37e-12 | 7.6907 (1.48e-14 / 3.50e-14) | 75.90% / 0.6925 / 0.6147 |
| Input SNR: 2.5 dB | PESQ (207) | 185 / 22 / 0 | +0.4343 / +0.4391 / +0.4293 | 2.76e-33 / 4.13e-32 / 7.58e-33 | 11.5646 (6.23e-31 / 1.18e-29) | 89.37% / 0.9271 / 1.2359 |
| Input SNR: 2.5 dB | STOI (207) | 183 / 23 / 1 | +0.0283 / +0.0182 / +0.0255 | 3.97e-32 / 5.56e-31 / 9.71e-32 | 10.7551 (5.61e-27 / 8.97e-26) | 88.41% / 0.8643 / 0.8398 |
| Input SNR: 7.5 dB | PESQ (206) | 191 / 15 / 0 | +0.4444 / +0.4376 / +0.4547 | 4.88e-40 / 9.28e-39 / 2.69e-39 | 11.3913 (4.62e-30 / 8.32e-29) | 92.72% / 0.9154 / 1.2777 |
| Input SNR: 7.5 dB | STOI (206) | 176 / 30 / 0 | +0.0214 / +0.0151 / +0.0174 | 2.49e-26 / 2.74e-25 / 4.57e-26 | 10.2216 (1.59e-24 / 1.90e-23) | 85.44% / 0.8214 / 0.7651 |
| Input SNR: 12.5 dB | PESQ (207) | 186 / 21 / 0 | +0.4032 / +0.4265 / +0.4215 | 3.24e-34 / 5.18e-33 / 1.02e-33 | 10.6409 (1.92e-26 / 2.88e-25) | 89.86% / 0.8530 / 1.0871 |
| Input SNR: 12.5 dB | STOI (207) | 169 / 36 / 2 | +0.0138 / +0.0064 / +0.0097 | 8.43e-22 / 5.06e-21 / 1.09e-21 | 9.2463 (2.32e-20 / 1.86e-19) | 81.64% / 0.7448 / 0.6102 |
| Input SNR: 17.5 dB | PESQ (204) | 186 / 18 / 0 | +0.3078 / +0.3096 / +0.3197 | 2.32e-36 / 3.95e-35 / 8.52e-36 | 10.5869 (3.43e-26 / 4.80e-25) | 91.18% / 0.8549 / 0.9298 |
| Input SNR: 17.5 dB | STOI (204) | 141 / 56 / 7 | +0.0081 / +0.0023 / +0.0042 | 1.21e-09 / 1.21e-09 / 1.21e-09 | 6.8696 (6.48e-12 / 6.48e-12) | 69.12% / 0.5645 / 0.3909 |

## 6. Exploratory Operational Thresholds, Non-Overlapping Categories & Sensitivity Analysis

**Literature Basis & Exploratory Operational Framing (Methodological Correction 3):** Neither ITU-T Rec. P.862.2 (2007) nor Taal et al. (2011) prescribes a single universal clinical JND/MCID constant for per-utterance Delta PESQ or Delta STOI. Instead, ITU-T Rec. P.862.2 (Section 6 / Table 1), ITU-T Rec. P.800.1, Hu & Loizou (2008, IEEE TASLP 16(1):229-238), and Loizou (2013, Ch. 11) document that objective Wideband PESQ predictions exhibit a residual root-mean-square error of ~0.10–0.25 MOS-LQO relative to human subjective MOS panels, while Taal et al. (2011, IEEE TASLP 19(7):2125-2136) demonstrate a nonlinear logistic mapping I(STOI) = 100 / (1 + exp(a * STOI + b)) that saturates near 100% word intelligibility when STOI > 0.90. Accordingly, all categorical cutoffs below are strictly designated as Exploratory Operational Thresholds, paired with a multi-threshold sensitivity analysis.

### A. Predefined Non-Overlapping Operational Categories (Summing to Exact N and 100.00%)

| Metric & Regime (N) | 1. Substantial Improvement | 2. Moderate Improvement | 3. Operational Equivalence (ROPE) | 4. Moderate Degradation | 5. Substantial Degradation | Total (Net Imp / Equiv / Net Deg) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Wideband PESQ Pooled (N=824, tau=0.10/0.35) | d >= +0.3500: 472 (57.28%) | +0.1000 <= d < +0.3500: 227 (27.55%) | -0.1000 < d < +0.1000: 78 (9.47%) | -0.3500 < d <= -0.1000: 25 (3.03%) | d <= -0.3500: 22 (2.67%) | 824 (84.83% / 9.47% / 5.70%) |
| STOI Pooled (N=824, tau=0.01/0.03) | d >= +0.0300: 204 (24.76%) | +0.0100 <= d < +0.0300: 184 (22.33%) | -0.0100 < d < +0.0100: 396 (48.06%) | -0.0300 < d <= -0.0100: 32 (3.88%) | d <= -0.0300: 8 (0.97%) | 824 (47.09% / 48.06% / 4.85%) |
| STOI Low-SNR 2.5 & 7.5 dB (N=413, tau=0.02/0.04) | d >= +0.0400: 102 (24.70%) | +0.0200 <= d < +0.0400: 86 (20.82%) | -0.0200 < d < +0.0200: 214 (51.82%) | -0.0400 < d <= -0.0200: 6 (1.45%) | d <= -0.0400: 5 (1.21%) | 413 (45.52% / 51.82% / 2.66%) |
| STOI High-SNR 12.5 & 17.5 dB (N=411, tau=0.005/0.015) | d >= +0.0150: 103 (25.06%) | +0.0050 <= d < +0.0150: 83 (20.19%) | -0.0050 < d < +0.0050: 189 (45.99%) | -0.0150 < d <= -0.0050: 24 (5.84%) | d <= -0.0150: 12 (2.92%) | 411 (45.26% / 45.99% / 8.76%) |

### B. Multi-Threshold Sensitivity Analysis Grid (FT vs. Zero-Shot, N = 824)

| Metric | Operational Cutoff tau | Improved (d >= +tau) | Equivalent (-tau < d < +tau) | Degraded (d <= -tau) | Row Sum Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Wideband PESQ | tau = 0.05 MOS-LQO | 724 (87.86%) | 40 (4.85%) | 60 (7.28%) | 824 (100.00%) |
| Wideband PESQ | tau = 0.10 MOS-LQO | 699 (84.83%) | 78 (9.47%) | 47 (5.70%) | 824 (100.00%) |
| Wideband PESQ | tau = 0.15 MOS-LQO | 668 (81.07%) | 120 (14.56%) | 36 (4.37%) | 824 (100.00%) |
| Wideband PESQ | tau = 0.20 MOS-LQO | 629 (76.33%) | 165 (20.02%) | 30 (3.64%) | 824 (100.00%) |
| Wideband PESQ | tau = 0.25 MOS-LQO | 579 (70.27%) | 218 (26.46%) | 27 (3.28%) | 824 (100.00%) |
| Wideband PESQ | tau = 0.35 MOS-LQO | 472 (57.28%) | 330 (40.05%) | 22 (2.67%) | 824 (100.00%) |
| Wideband PESQ | tau = 0.50 MOS-LQO | 305 (37.01%) | 501 (60.80%) | 18 (2.18%) | 824 (100.00%) |
| STOI | tau = 0.005 (0.5%) | 470 (57.04%) | 294 (35.68%) | 60 (7.28%) | 824 (100.00%) |
| STOI | tau = 0.010 (1.0%) | 388 (47.09%) | 396 (48.06%) | 40 (4.85%) | 824 (100.00%) |
| STOI | tau = 0.015 (1.5%) | 321 (38.96%) | 478 (58.01%) | 25 (3.03%) | 824 (100.00%) |
| STOI | tau = 0.020 (2.0%) | 272 (33.01%) | 536 (65.05%) | 16 (1.94%) | 824 (100.00%) |
| STOI | tau = 0.030 (3.0%) | 204 (24.76%) | 612 (74.27%) | 8 (0.97%) | 824 (100.00%) |
| STOI | tau = 0.050 (5.0%) | 98 (11.89%) | 723 (87.74%) | 3 (0.36%) | 824 (100.00%) |

## 7. Methodological Limitations & Separate Output Directory Inventory

- **Limitation 1 — Two-Speaker Benchmark Clustering:** The official Edinburgh VoiceBank-DEMAND test set comprises 824 utterances from only 2 speakers (p232 and p257). While speaker-stratified bootstrapping (B = 10,000) and separate speaker tests prove that improvements are highly consistent across both individuals, inference is scoped strictly to this benchmark and cannot estimate random-effects variance across a broad human speaker population.
- **Limitation 2 — STOI Ceiling Compression at High SNRs:** At 12.5 dB and 17.5 dB input SNR, baseline zero-shot STOI is already 0.9444 and 0.9594 (medians 0.9569 and 0.9787), compressing the maximum possible positive STOI gain and inducing right-skewness (g_1 = +1.4740) in paired STOI differences. While statistically significant (Sign Test p_Holm = 3.72e-81), 48.06% of utterances fall within the narrow [-0.01, +0.01] STOI operational equivalence band, demonstrating why statistical significance (p < 0.05) must be distinguished from practical intelligibility change.
- **Limitation 3 — Objective Surrogates vs. Human Listening Panels:** ITU-T P.862.2 Wideband PESQ and STOI are validated objective computational surrogates for perceptual quality and intelligibility, not direct human MOS / word-recognition listening tests.
- **Separate Statistical Artifacts Generated in results/statistical_analysis/:**
1. `results/statistical_analysis/statistical_analysis_full_report.json (207,209 bytes)`
2. `results/statistical_analysis/primary_and_secondary_tests.csv (2,870 bytes)`
3. `results/statistical_analysis/exploratory_subgroup_tests.csv (6,268 bytes)`
4. `results/statistical_analysis/symmetry_and_normality_diagnostics.csv (4,992 bytes)`
5. `results/statistical_analysis/operational_threshold_categories.csv (2,212 bytes)`
6. `results/statistical_analysis/operational_threshold_sensitivity.csv (614 bytes)`
7. `results/statistical_analysis/statistical_diagnostics_and_effect_sizes.svg (67,674 bytes)`
