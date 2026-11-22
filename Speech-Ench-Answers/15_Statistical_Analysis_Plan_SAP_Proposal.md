> **Document ID:** `15` | **Phase:** Phase 4D-Plan — Statistical Analysis Plan (SAP) | **Role:** Statistical Protocol Proposal  
> **Source Word Document:** `Statistical_Analysis_Plan_SAP_Proposal.docx` | **Repository Path:** `Speech-Ench-Answers/15_Statistical_Analysis_Plan_SAP_Proposal.md`

---
# Proposed Statistical Analysis Plan (SAP): Paired Hypothesis Testing & Practical Effect-Size Protocol

**Prepared By:** Research / QA Engineer Subagent (qa-research-engineer, Model: Pro, Workspace: inherit, Write access: DISABLED, Strictly read-only)

## Executive Status & Stage 1 Compliance

- **Current Stage: Stage 1 — Pre-Execution Statistical Analysis Plan (SAP) Proposal (Awaiting Approval)**
- **Tests Executed / p-Values Claimed:** 0 (No hypothesis tests have been executed yet; no statistical significance claims are made prior to execution).
- **Artifact & Workspace Preservation:** 100% Read-Only. Zero files in the project workspace were modified, created, or overwritten, and no new experiments were run.

## 1. Verified Data Sources, Pairing Architecture & Precision

- **Inspected Read-Only Sources:**
  - `results/test_baseline_per_utterance.csv:` 2,472 data rows (824 utterances x 3 baseline conditions: noisy, spectral_gating, dns48_zeroshot), 0 missing or failed rows.
  - `results/test_finetuned_per_utterance.csv:` 824 data rows for dns48_finetuned (best_val_pesq.th, Epoch 14), 0 missing or failed rows.
  - `manifests/test_manifest.json:` Authoritative metadata for all N = 824 test utterances (speakers p232: 393 utterances; p257: 431 utterances; 5 unseen noise types; 4 unseen SNR levels).
- **Deterministic Pairing Protocol:**
  - Exact 1:1 inner join on utterance_id sorted lexicographically (p232_001 through p257_434), asserting identical speaker_id, noise_type, snr_db, and num_samples across paired records.
  - For each utterance i in {1, ..., 824} and metric M in {PESQ_wb, STOI}, the paired difference is defined as: d_i = x_{i,FT} - x_{i,ZS}.
  - **Rounding Precision Note:** Because pesq_wb and stoi in the per-utterance CSVs are stored to 4 decimal places (round(float(score), 4) in src/evaluation/evaluator.py), exact zero differences (d_i = 0) and tied absolute differences (|d_i| = |d_j|) can occur and are explicitly accounted for in Section 3D.

## 2. Hypothesis Families & Experimental Comparisons

### A. Family 1 — Co-Primary Hypotheses (m = 2 tests)

Evaluating Demucs dns48 Fine-Tuned (Epoch 14, best_val_pesq.th) vs. Demucs dns48 Zero-Shot across all N = 824 paired test utterances:

1. **H_{0,PESQ}:** The distribution of paired Wideband PESQ differences d_{i,PESQ} = PESQ_{i,FT} - PESQ_{i,ZS} is centered at zero location shift (theta_{PESQ} = 0) vs. **H_{1,PESQ}:** theta_{PESQ} != 0 (two-sided).
2. **H_{0,STOI}:** The distribution of paired STOI differences d_{i,STOI} = STOI_{i,FT} - STOI_{i,ZS} is centered at zero location shift (theta_{STOI} = 0) vs. **H_{1,STOI}:** theta_{STOI} != 0 (two-sided).

### B. Family 2 — Secondary Baseline Context Comparisons (m = 4 tests)

- **Fine-Tuned dns48 vs. Unprocessed Noisy Audio** (N = 824, Wideband PESQ & STOI).
- **Fine-Tuned dns48 vs. Stationary Spectral Gating** (N = 824, Wideband PESQ & STOI).

### C. Family 3 — Stratified Subgroup Comparisons (m = 22 tests: Fine-Tuned vs. Zero-Shot)

- **By Speaker (2 x 2 = 4 tests):** p232 (n = 393), p257 (n = 431).
- **By Unseen Noise Type (5 x 2 = 10 tests):** bus (n = 164), cafe (n = 164), living (n = 166), office (n = 164), psquare (n = 166).
- **By Input SNR Level (4 x 2 = 8 tests):** 2.5 dB (n = 207), 7.5 dB (n = 206), 12.5 dB (n = 207), 17.5 dB (n = 204).

## 3. Suitability, Assumptions & Limitations of Paired Non-Parametric Testing

### A. Why a Paired Non-Parametric Test (Wilcoxon Signed-Rank Test) Is Appropriate

1. **Paired Observation Structure:** Every test utterance i has a fixed speaker, linguistic content, room/noise realization, and SNR, creating strong positive covariance Cov(X_{i,FT}, X_{i,ZS}) > 0. Paired differencing eliminates between-utterance variance.
2. **Bounded, Non-Gaussian Metric Scales:** Wideband PESQ (ITU-T P.862.2) is bounded in [1.0, 4.64] via a non-linear logistic mapping to MOS-LQO. STOI is bounded in [0, 1] and exhibits strong ceiling compression (and negative skewness) on VoiceBank-DEMAND because many 12.5 dB and 17.5 dB utterances already have baseline STOI > 0.94. The two-sided Wilcoxon signed-rank test does not assume Gaussianity and is far more robust to outliers and ceiling compression than parametric tests.

### B. Formal Assumptions of the Wilcoxon Signed-Rank Test & Pre-Test Diagnostics

- **Continuous Paired Differences:** d_i = x_{i,FT} - x_{i,ZS} are continuous interval-scale differences (subject to 4-decimal CSV quantization).
- **Symmetry of d_i Around the Pseudo-Median (theta_HL):** The Wilcoxon signed-rank test formally evaluates whether the distribution of d_i is symmetric around theta = 0 (specifically testing whether the Hodges–Lehmann pseudo-median theta_HL = median{(d_i + d_j)/2 : i <= j} equals zero). If d_i is skewed (e.g., for STOI due to ceiling effects near 1.0), the pseudo-median theta_HL can differ slightly from the sample median Median(d_i).
- **Pre-Test Diagnostics to Execute:** Compute sample skewness (gamma_1), excess kurtosis (gamma_2), Shapiro–Wilk test, and D'Agostino's K^2 test on d_i.
- **Distribution-Free Robustness Check (Exact Paired Sign Test):** Because the exact two-sided Paired Sign Test (S = N^+ ~ Binomial(N_nonzero, 0.5) under H_0: Median(d_i) = 0) requires neither normality nor symmetry of d_i, we will execute it alongside the Wilcoxon signed-rank test.
- **Parametric Reference (Paired Student's t-test):** At N = 824, the Central Limit Theorem ensures asymptotic normality of the sample mean difference d_bar; we will report the two-sided paired t-test as a secondary parametric reference.

### C. Key Structural Limitation: Clustered / Hierarchical Design (2 Test Speakers)

- **Limitation:** Although N = 824 utterances are distinct recordings, they originate from only 2 unseen test speakers (p232 with 393 utterances; p257 with 431 utterances). Utterances from the same speaker share vocal-tract acoustics, pitch range, and articulation patterns, meaning d_1, ..., d_824 are clustered by speaker rather than strictly i.i.d. drawings from the global human speaker population.
- **Scope of Inference & Remediation:** (1) Inference on pooled N = 824 formally applies to the official Edinburgh VoiceBank-DEMAND test benchmark population (utterances from speakers p232 and p257 across the 20 unseen noise x SNR conditions). (2) To verify that results are not driven by a single speaker or intra-speaker correlation, we will run separate Wilcoxon signed-rank and sign tests within each speaker cluster (p232 and p257 independently) and compute speaker-stratified bootstrap 95% confidence intervals (B = 10,000 resamples, resampling utterances independently within p232 and p257).

### D. Handling of Zero Differences (d_i = 0) and Tied Ranks (|d_i| = |d_j|)

- **Primary Zero Method:** Standard Wilcoxon zero exclusion (zero_method="wilcox"), where exact ties d_i = 0 are counted and reported as N^0, and ranking is performed on the N_nonzero = N - N^0 non-zero differences.
- **Sensitivity Check:** Pratt's zero method (zero_method="pratt", which includes zero differences in ranking before dropping their zero-signed ranks) will also be computed to confirm invariance to 4-decimal rounding zeros.
- **Tie & Continuity Correction:** Average fractional ranks will be assigned to tied |d_i| values, and the standardized Z-statistic will use the exact tie-corrected variance formula Var(W^+) = [N_nz(N_nz+1)(2N_nz+1) - 0.5 * sum(t_k(t_k-1)(t_k+1))] / 24.

## 4. Effect Sizes & 95% Confidence Intervals

For every comparison, both non-parametric and parametric effect sizes will be calculated with 95% Confidence Intervals:

| Effect Size / Estimator | Mathematical Definition | Confidence Interval Method | Purpose / Interpretation |
| :--- | :--- | :--- | :--- |
| Hodges–Lehmann Shift (theta_HL) | median_{1 <= i <= j <= N_nz} ((d_i + d_j) / 2) | Non-parametric Wilcoxon inversion & BCa Bootstrap 95% CI (B=10,000) | Canonical location-shift parameter estimated by the Wilcoxon signed-rank test (native PESQ/STOI units) |
| Sample Median & IQR | Q_50(d_i), IQR = Q_75(d_i) - Q_25(d_i) | Order-statistic & Stratified Bootstrap 95% CI | Symmetry-free median shift and robust interquartile dispersion |
| Matched-Pairs Rank-Biserial (r_rb) | r_rb = (W^+ - W^-) / (W^+ + W^-) in [-1, +1] | Bootstrap 95% CI (B=10,000) | Natural rank-based effect size for Wilcoxon signed-rank test (net dominance of positive vs. negative ranks) |
| Standardized Wilcoxon (r) | r = \|Z\| / sqrt(N) | Bootstrap 95% CI (B=10,000) | Standardized non-parametric effect size (0.1 small, 0.3 medium, 0.5 large) |
| Probability of Superiority (CLES / Win Rate) | (N^+ + 0.5 * N^0) / N and strict win rate N^+ / N | Wilson Score & Clopper–Pearson 95% CI | Probability that Fine-Tuned outperforms Zero-Shot on a randomly chosen test utterance |
| Sample Mean Shift (d_bar) & Paired Cohen's d_z (Hedges' g_z) | d_bar = mean(d_i), d_z = d_bar / s_d | Analytic t-interval & Speaker-Stratified Bootstrap 95% CI | Parametric mean gain and standardized paired effect size |

## 5. Multiple-Testing Correction Strategy

To control the Family-Wise Error Rate (FWER) at alpha = 0.05 (two-sided):

1. **Family 1 — Co-Primary Tests (m = 2: FT vs. ZS on PESQ and STOI):** Holm–Bonferroni step-down procedure. Sort p_(1) <= p_(2). Test p_(1) against alpha/2 = 0.025; if rejected, test p_(2) against alpha/1 = 0.050. Both raw unadjusted p-values and Holm-adjusted p_Holm will be reported.
2. **Family 2 — Secondary Baseline Comparisons (m = 4: FT vs. Noisy and FT vs. Spectral Gating on PESQ & STOI):** Holm–Bonferroni step-down procedure across the 4 tests (alpha/4 = 0.0125, alpha/3 = 0.0167, alpha/2 = 0.0250, alpha/1 = 0.0500).
3. **Family 3 — Subgroup Stratifications (m = 22 tests across Speaker, Unseen Noise, and SNR):** Report unadjusted p-values, Holm–Bonferroni FWER-adjusted p_Holm across all 22 tests, and Benjamini–Hochberg FDR-adjusted q_BH at q = 0.05.

## 6. Distinguishing Statistical Significance from Practical Audio-Quality Improvement

Because N = 824 provides very high statistical power, even an inaudible shift (e.g., Delta PESQ = +0.02 or Delta STOI = +0.001) could yield p < 0.001. We therefore separate inferential statistical significance (p_Holm < 0.05) from practical perceptual significance using literature-backed Just Noticeable Difference (JND) / Minimum Clinically Important Difference (MCID) thresholds:

| Metric | Practical Threshold Tier | Criterion | Perceptual / Acoustic Interpretation |
| :--- | :--- | :--- | :--- |
| Wideband PESQ | Region of Practical Equivalence (ROPE) | \|Delta PESQ\| < 0.10 | Imperceptible or negligible difference in listening quality |
| Wideband PESQ | Noticeable Perceptual Improvement (JND) | Delta PESQ >= +0.10 (and >= +0.20) | Audible improvement in speech quality (ITU-T P.862 / Loizou) |
| Wideband PESQ | Substantial Perceptual Improvement | Delta PESQ >= +0.35 (and >= +0.50) | Clear to major perceptual quality improvement |
| Wideband PESQ | Noticeable Perceptual Degradation | Delta PESQ <= -0.10 | Audible quality degradation |
| STOI | Region of Practical Equivalence (ROPE) | \|Delta STOI\| < 0.01 (1.0%) | Negligible intelligibility index change |
| STOI | Noticeable Intelligibility Gain | Delta STOI >= +0.01 (high-STOI) / >= +0.02 (low-SNR) | Meaningful intelligibility gain accounting for ceiling compression > 0.90 |
| STOI | Noticeable Intelligibility Degradation | Delta STOI <= -0.01 (-1.0%) | Meaningful intelligibility reduction |

- **Practical Significance Evaluation Criteria:**
  - Verify whether the lower bound of the 95% Confidence Interval for the location shift (theta_HL and d_bar) exceeds the practical JND threshold.
  - Report exact counts and percentages of the 824 test utterances falling into each practical category (Substantial Improvement, Noticeable Improvement, Practical Equivalence / ROPE, Noticeable Degradation).

## 7. Artifact Preservation & Execution Guarantees

- **Zero Modification of Existing Artifacts:** All Phase 3A, 3B, 3C, 4, and 4C files (manifests/, checkpoints/, results/test_baseline_*, results/test_finetuned_*, results/final_test_comparison.*, results/test_breakdown_*) will be read strictly in read-only mode and preserved untouched.
- **No Retraining or Model Changes:** No test data will be used to modify any model, hyperparameter, checkpoint, or evaluation procedure.
- **Next Step:** Awaiting user approval of this Statistical Analysis Plan (SAP) before executing the hypothesis tests.
