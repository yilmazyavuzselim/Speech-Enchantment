# Speech-Enhancement

Academic speech enhancement system reducing background noise while preserving perceptual speech quality and intelligibility on the **University of Edinburgh Noisy Speech Database (VoiceBank-DEMAND)**.

---

## Project Overview

- **Dataset:** University of Edinburgh Noisy Speech Database ([DOI: 10.7488/ds/2117](https://doi.org/10.7488/ds/2117), Handle: `10283/2791`, 28-speaker VoiceBank-DEMAND), licensed under **Creative Commons Attribution 4.0 International (`CC BY 4.0`)** and resampled from 48 kHz to **16 kHz, 16-bit PCM mono**.
- **Main Pretrained Model:** Facebook Research Causal Demucs (`dns48` checkpoint, `hidden=48`, 18,867,937 parameters, 16 kHz 1D waveform U-Net + LSTM).
- **Traditional DSP Baseline:** Stationary Spectral Gating (`noisereduce==3.0.2`, `n_fft=512, win_length=512, hop_length=128, n_std_thresh_stationary=1.5, prop_decrease=1.0`).
- **Fine-Tuning Loss Function:** Time-domain $L_1$ loss + Multi-Resolution STFT loss ($\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_1 + 0.1 \cdot \mathcal{L}_{\text{SC}} + 0.1 \cdot \mathcal{L}_{\text{Mag}}$, weights applied once in the trainer across FFT sizes `[1024, 2048, 512]`).
- **Evaluation Metrics:** ITU-T P.862.2 Wideband PESQ (`16 kHz`, `'wb'`) and Short-Time Objective Intelligibility (STOI, `16 kHz`, `extended=False`) across all 824 official test utterances and 770 held-out validation utterances.

---

## Dataset Splits & Verification Statistics

Full dataset integrity verification ([results/dataset_validation_report.json](results/dataset_validation_report.json)) completed with **0 anomalies** across all 12,396 clean/noisy utterance pairs (24,792 resampled 16 kHz WAV files).

| Split | Speakers (Count) | Utterances | Duration | Noise Types | SNR (dB) Levels |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 26 speakers (`p227`, `p228`, `p230`, `p231`, `p233`, `p236`, `p239`, `p243`, `p244`, `p250`, `p254`, `p256`, `p258`, `p259`, `p267`, `p268`, `p269`, `p270`, `p273`, `p274`, `p276`, `p277`, `p278`, `p279`, `p282`, `p286`) | 10,802 (12,272 chunks @ 4.0s / 2.0s stride) | 8.758 hours | 10 DEMAND noises (`babble`, `cafeteria`, `car`, `kitchen`, `meeting`, `metro`, `restaurant`, `ssn`, `station`, `traffic`) | 0.0, 5.0, 10.0, 15.0 |
| **Validation** | 2 speakers (`p226`, `p287`) | 770 (full-length uncropped) | 0.633 hours (38.0 min) | 10 DEMAND noises (same as train) | 0.0, 5.0, 10.0, 15.0 |
| **Test** | 2 speakers (`p232`, `p257`) | 824 (full-length uncropped) | 0.576 hours (34.5 min) | 5 unseen DEMAND noises (`bus`, `cafe`, `living`, `office`, `psquare`) | 2.5, 7.5, 12.5, 17.5 |
| **Total** | **30 unique speakers** | **12,396 pairs** | **9.967 hours** | **15 distinct noise types** | **Zero anomalies** |

> [!NOTE]
> **Strict Speaker Isolation:** Test speakers (`p232` and `p257`) have **zero overlap** with the training or validation sets (`overlap: []`). Validation speakers (`p226` and `p287`) are held out completely from model parameter updates and used exclusively for checkpoint selection.

---

## Baseline Benchmark Results (Phase 3A)

Evaluated across all 824 official test utterances ([results/test_baseline_summary.json](results/test_baseline_summary.json)) and all 770 validation utterances ([results/val_baseline_summary.json](results/val_baseline_summary.json)) with **0 failed evaluations across all 1,594 audio pairs**.

### 1. Official Test Set Baselines (824 Utterances, Unseen Noise Types & Speakers)

| Condition | PESQ (Mean ± Std) | PESQ Median | STOI (Mean ± Std) | STOI Median | $\Delta$ PESQ vs Noisy | $\Delta$ STOI vs Noisy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Noisy Audio (Unprocessed)** | 1.9719 ± 0.7532 | 1.7986 | 0.9211 ± 0.0709 | 0.9403 | — | — |
| **Stationary Spectral Gating** | 1.4232 ± 0.2848 | 1.3502 | 0.8234 ± 0.0655 | 0.8320 | -0.5487 | -0.0977 |
| **Pretrained Demucs `dns48` (Zero-Shot)** | **2.5203 ± 0.6258** | **2.5263** | **0.9300 ± 0.0638** | **0.9476** | **+0.5484** | **+0.0089** |

### 2. Validation Set Baselines (770 Utterances, Held-Out Speakers `p226` and `p287`)

| Condition | PESQ (Mean ± Std) | PESQ Median | STOI (Mean ± Std) | STOI Median | $\Delta$ PESQ vs Noisy | $\Delta$ STOI vs Noisy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Noisy Audio (Unprocessed)** | 1.5714 ± 0.5350 | 1.4186 | 0.8126 ± 0.1118 | 0.8286 | — | — |
| **Stationary Spectral Gating** | 1.2744 ± 0.1960 | 1.2239 | 0.7392 ± 0.0990 | 0.7589 | -0.2970 | -0.0734 |
| **Pretrained Demucs `dns48` (Zero-Shot)** | **2.0691 ± 0.6258** | **2.0111** | **0.8523 ± 0.0964** | **0.8691** | **+0.4977** | **+0.0397** |

---

## Fine-Tuning Results (Phase 3B — Demucs `dns48`)

Fine-tuning was conducted on a Google Colab **NVIDIA L4 GPU** across 15 full epochs (`4,123.87s` / `68.73 min` total wall-clock runtime). Model selection was performed strictly on the 770 held-out validation utterances (`p226` and `p287`) without accessing the official test set.

### 1. Training Configuration & Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | Demucs `dns48` | Causal waveform U-Net + LSTM (`hidden=48`, 16 kHz, 18,867,937 parameters) |
| **Training Dataset** | 10,802 utterances | 12,272 deterministic sliding-window chunks (4.0s segment, 2.0s stride) across 26 train speakers |
| **Batch Size** | 16 | 767 batches per epoch (`12,272 / 16`) |
| **Optimizer** | Adam | $\text{lr} = 5 \times 10^{-5}$, $\beta = (0.9, 0.999)$, $\text{weight\_decay} = 0.0$ |
| **Gradient Clipping** | 5.0 | Max $L_2$-norm gradient clipping (`torch.nn.utils.clip_grad_norm_`) |
| **Loss Function** | Multi-Resolution STFT + $L_1$ | $\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_1 + 0.1 \cdot \mathcal{L}_{\text{SC}} + 0.1 \cdot \mathcal{L}_{\text{Mag}}$ |
| **STFT Resolutions** | 3 filter banks | FFT sizes: `[1024, 2048, 512]`, Hops: `[120, 240, 50]`, Windows: `[600, 1200, 240]` |
| **Training Epochs** | 15 full epochs | Atomic checkpointing per epoch (`best_val_pesq.th`, `best_val_loss.th`, `checkpoint.th`) |
| **Hardware** | NVIDIA L4 GPU (24 GB) | Steady-state: ~270s/epoch, Peak GPU allocated memory: `4,189.64 MB` |

### 2. Validation Milestones (770 Held-Out Validation Utterances)

| Model / Condition | Val Loss | PESQ (Mean ± Std) | PESQ Median | STOI (Mean ± Std) | STOI Median | $\Delta$ PESQ vs Noisy | $\Delta$ PESQ vs Zero-Shot |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Noisy Audio (Unprocessed)** | — | 1.5714 ± 0.5350 | 1.4186 | 0.8126 ± 0.1118 | 0.8286 | — | — |
| **Stationary Spectral Gating** | — | 1.2744 ± 0.1960 | 1.2239 | 0.7392 ± 0.0990 | 0.7589 | -0.2970 | -0.7947 |
| **Pretrained Demucs `dns48` (Epoch 00)** | 0.1342 | 2.0691 ± 0.6258 | 2.0111 | 0.8523 ± 0.0964 | 0.8691 | +0.4977 | — |
| **Fine-Tuned Demucs (`best_val_loss.th`, Ep 9)** | **0.0919** | 2.5614 ± 0.6629 | 2.5648 | **0.8844 ± 0.0764** | **0.8989** | +0.9900 | +0.4923 |
| **Fine-Tuned Demucs (`best_val_pesq.th`, Ep 14)** | 0.0920 | **2.5691 ± 0.6668** | **2.5843** | 0.8835 ± 0.0784 | 0.8981 | **+0.9977** | **+0.5000** |
| **Fine-Tuned Demucs (`checkpoint.th`, Ep 15)** | 0.0924 | 2.5646 ± 0.6749 | 2.5697 | 0.8831 ± 0.0791 | 0.8966 | +0.9932 | +0.4955 |

Full epoch-by-epoch logs are recorded in [results/training_history.csv](results/training_history.csv) and [results/training_summary.json](results/training_summary.json), and plotted in [results/training_curves.png](results/training_curves.png).

---

## Official Test-Set Evaluation & 4-Way Comparative Results (Phase 4)

The primary checkpoint selected via validation PESQ (`best_val_pesq.th`, Epoch 14) was evaluated once across all **824 official Edinburgh VoiceBank-DEMAND test utterances** (speakers `p232` and `p257`, 5 unseen DEMAND noise types, 4 unseen SNR levels) and compared 1:1 against the Phase 3A baselines ([results/final_test_comparison.csv](results/final_test_comparison.csv), [results/final_test_comparison.json](results/final_test_comparison.json), [results/final_test_comparison.png](results/final_test_comparison.png)).

### 1. Overall Performance Comparison (824 Test Utterances)

| Condition / Model | PESQ (Mean ± Std) | PESQ Median | STOI (Mean ± Std) | STOI Median | $\Delta$ PESQ vs Noisy | $\Delta$ PESQ vs Zero-Shot | $\Delta$ STOI vs Noisy | $\Delta$ STOI vs Zero-Shot | Successful / Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Noisy Audio (Unprocessed)** | 1.9719 ± 0.7532 | 1.7986 | 0.9211 ± 0.0709 | 0.9403 | — | -0.5484 (-21.76%) | — | -0.0089 (-0.96%) | 824 / 824 |
| **Stationary Spectral Gating** | 1.4232 ± 0.2848 | 1.3502 | 0.8234 ± 0.0655 | 0.8320 | -0.5487 (-27.83%) | -1.0971 (-43.53%) | -0.0977 (-10.61%) | -0.1066 (-11.46%) | 824 / 824 |
| **Pretrained Demucs `dns48` (Zero-Shot)** | 2.5203 ± 0.6258 | 2.5263 | 0.9300 ± 0.0638 | 0.9476 | +0.5484 (+27.81%) | — | +0.0089 (+0.97%) | — | 824 / 824 |
| **Fine-Tuned Demucs `dns48` (Epoch 14)** | **2.9180 ± 0.6483** | **3.0316** | **0.9480 ± 0.0507** | **0.9645** | **+0.9461 (+47.98%)** | **+0.3977 (+15.78%)** | **+0.0269 (+2.92%)** | **+0.0180 (+1.94%)** | **824 / 824** |

### 2. Performance Breakdown by Test Input SNR Level ([results/test_breakdown_by_snr.csv](results/test_breakdown_by_snr.csv))

| SNR Level | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs `dns48` Zero-Shot PESQ | Fine-Tuned Demucs PESQ | $\Delta$PESQ vs Zero-Shot | Noisy STOI | Demucs `dns48` Zero-Shot STOI | Fine-Tuned Demucs STOI | $\Delta$STOI vs Zero-Shot |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2.5 dB** | 207 | 1.4232 ± 0.4251 | 1.2810 ± 0.2183 | 1.9860 ± 0.4584 | **2.4203 ± 0.6152** | **+0.4343** | 0.8683 ± 0.0864 | 0.8907 ± 0.0748 | **0.9190 ± 0.0604** | **+0.0283** |
| **7.5 dB** | 206 | 1.7651 ± 0.5921 | 1.3968 ± 0.2501 | 2.3774 ± 0.5122 | **2.8218 ± 0.5718** | **+0.4444** | 0.9188 ± 0.0590 | 0.9260 ± 0.0573 | **0.9474 ± 0.0444** | **+0.0214** |
| **12.5 dB** | 207 | 2.1041 ± 0.6947 | 1.4446 ± 0.2574 | 2.6599 ± 0.5211 | **3.0631 ± 0.5385** | **+0.4032** | 0.9403 ± 0.0501 | 0.9444 ± 0.0500 | **0.9582 ± 0.0398** | **+0.0138** |
| **17.5 dB** | 204 | 2.6035 ± 0.7063 | 1.5722 ± 0.3240 | 3.0651 ± 0.4485 | **3.3728 ± 0.4476** | **+0.3078** | 0.9574 ± 0.0464 | 0.9594 ± 0.0478 | **0.9674 ± 0.0416** | **+0.0081** |

### 3. Performance Breakdown by Unseen DEMAND Noise Type ([results/test_breakdown_by_noise.csv](results/test_breakdown_by_noise.csv))

| Noise Type | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs `dns48` Zero-Shot PESQ | Fine-Tuned Demucs PESQ | $\Delta$PESQ vs Zero-Shot | $\Delta$PESQ vs Noisy | Fine-Tuned Demucs STOI | $\Delta$STOI vs Zero-Shot |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **office** | 164 | 2.5361 ± 0.7057 | 1.5682 ± 0.3317 | 2.9006 ± 0.5518 | **3.4342 ± 0.4769** | **+0.5337** | **+0.8982** | **0.9667 ± 0.0464** | **+0.0169** |
| **bus** | 164 | 2.4845 ± 0.7773 | 1.5220 ± 0.3188 | 2.8184 ± 0.5711 | **3.2107 ± 0.4532** | **+0.3922** | **+0.7262** | **0.9635 ± 0.0405** | **+0.0146** |
| **psquare** | 166 | 1.7367 ± 0.5673 | 1.3738 ± 0.2237 | 2.4527 ± 0.5435 | **2.8194 ± 0.5205** | **+0.3666** | **+1.0827** | **0.9430 ± 0.0490** | **+0.0165** |
| **cafe** | 164 | 1.4919 ± 0.4302 | 1.2904 ± 0.1989 | 2.1181 ± 0.5447 | **2.4668 ± 0.6115** | **+0.3487** | **+0.9749** | **0.9256 ± 0.0586** | **+0.0224** |
| **living** | 166 | 1.6178 ± 0.4898 | 1.3628 ± 0.2245 | 2.3150 ± 0.5432 | **2.6632 ± 0.6283** | **+0.3482** | **+1.0454** | **0.9411 ± 0.0455** | **+0.0193** |

### 4. Performance Breakdown by Unseen Test Speaker ([results/test_breakdown_by_speaker.csv](results/test_breakdown_by_speaker.csv))

| Speaker | Gender | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs `dns48` Zero-Shot PESQ | Fine-Tuned Demucs PESQ | $\Delta$PESQ vs Zero-Shot | Fine-Tuned Demucs STOI | $\Delta$STOI vs Zero-Shot |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **p232** | Male | 393 | 2.1687 ± 0.7147 | 1.3991 ± 0.2847 | 2.6535 ± 0.5909 | **3.0351 ± 0.5971** | **+0.3817** | **0.9504 ± 0.0504** | **+0.0104** |
| **p257** | Female | 431 | 1.7925 ± 0.7425 | 1.4451 ± 0.2831 | 2.3989 ± 0.6319 | **2.8112 ± 0.6742** | **+0.4123** | **0.9457 ± 0.0508** | **+0.0248** |

### 5. Representative Audio Demonstration Examples ([results/audio_examples/](results/audio_examples/))

Four representative test utterances covering all 4 SNR levels (`2.5`, `7.5`, `12.5`, `17.5` dB), 4 unseen noise environments, and both test speakers are included as complete 5-condition WAV sets (`clean.wav`, `noisy.wav`, `spectral_gating.wav`, `dns48_zeroshot.wav`, `dns48_finetuned.wav`):

| Utterance ID | Speaker | Noise Type | Input SNR | Duration | Noisy PESQ / STOI | Zero-Shot PESQ / STOI | Fine-Tuned PESQ / STOI | Improvement vs Zero-Shot | Audio Directory |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **p232_023** | `p232` (M) | `psquare` | 2.5 dB | 9.77s | 1.3108 / 0.9366 | 2.2626 / 0.9671 | **2.5008 / 0.9675** | **+0.2382 PESQ, +0.0004 STOI** | [results/audio_examples/p232_023/](results/audio_examples/p232_023/) |
| **p257_008** | `p257` (F) | `cafe` | 7.5 dB | 5.58s | 1.2496 / 0.9045 | 1.9051 / 0.9325 | **2.2039 / 0.9531** | **+0.2988 PESQ, +0.0206 STOI** | [results/audio_examples/p257_008/](results/audio_examples/p257_008/) |
| **p232_012** | `p232` (M) | `living` | 12.5 dB | 4.34s | 2.0530 / 0.9654 | 2.7081 / 0.9648 | **3.1346 / 0.9739** | **+0.4265 PESQ, +0.0091 STOI** | [results/audio_examples/p232_012/](results/audio_examples/p232_012/) |
| **p257_014** | `p257` (F) | `office` | 17.5 dB | 3.69s | 3.3299 / 0.9944 | 3.4156 / 0.9950 | **3.7012 / 0.9961** | **+0.2856 PESQ, +0.0011 STOI** | [results/audio_examples/p257_014/](results/audio_examples/p257_014/) |

- **Note on Smoke-Test Utterance `p232_001.wav` (`bus`, `17.5 dB` SNR):** Evaluated on the serialized 16 kHz 16-bit `PCM_16` test dataset at Noisy PESQ `2.9238` (STOI `0.8967`), Spectral Gating PESQ `1.6644` (STOI `0.8554`), Zero-Shot PESQ `3.0293` (STOI `0.9017`), and Fine-Tuned PESQ **`3.1112`** (STOI **`0.9094`**). The slight $\approx 0.0013\text{–}0.0021$ PESQ difference relative to the Step 2 single-pair smoke test (`2.9225` Noisy / `3.0272` Zero-Shot) arises because Step 2 resampled the 48 kHz WAVs in memory as unquantized 32-bit `float32` tensors without disk `PCM_16` quantization (`±1/32768`) or `[-1.0, 1.0]` clipping.

### 6. Interactive Offline Audio Demo & Portable Package (Phase 5B)

For interactive auditory inspection, an offline demonstration player and portable archive are provided directly within the repository:
- **Interactive Offline Web Demo:** [audio_demo/index.html](audio_demo/index.html) (see [audio_demo/README.md](audio_demo/README.md) for full guide).
- **Portable Standalone Package:** [Speech_Enhancement_Offline_Audio_Demo.zip](Speech_Enhancement_Offline_Audio_Demo.zip) (3.01 MB standalone archive with verified SHA-256 integrity manifest).

**Key Demonstration Capabilities:**
1. **100% Offline & Zero Dependencies:** Operates locally via `file://` protocol in modern web browsers with HTML5 audio support (automated verification performed in headless Microsoft Edge only) with zero installation, zero external CDN requests, and zero local web server requirements.
2. **Synchronized 5-Condition Audio Players:** Clean Reference, Noisy Mixture, Stationary Spectral Gating, Demucs `dns48` Zero-Shot, and Demucs `dns48` Fine-Tuned (Epoch 14), with single-audition mutual exclusion to prevent overlapping playback.
3. **Seamless A/B Comparison Controller:** Instantly toggles between any two selected conditions (e.g., Noisy vs. Fine-Tuned, or Zero-Shot vs. Fine-Tuned) while preserving the exact playback timestamp.
4. **Verified Cryptographic Integrity:** Contains 20 WAV files (3,741,710 total bytes) verified byte-for-byte identical to source evaluation outputs via [audio_demo/audio_integrity_manifest.json](audio_demo/audio_integrity_manifest.json).

**How to Audition:**
- **In-Repository:** Simply open `audio_demo/index.html` directly in your web browser.
- **Standalone Archive:** Extract `Speech_Enhancement_Offline_Audio_Demo.zip` to any local directory and open `index.html`.

---

## Statistical Hypothesis Testing & Effect-Size Analysis (Phase 4D)

A formal paired statistical evaluation ([results/statistical_analysis/](results/statistical_analysis/)) was conducted across all $N = 824$ paired test utterances ($d_i = x_{i,\text{FT}} - x_{i,\text{ref}}$), prioritizing the **exact two-sided Paired Sign Test** as the primary symmetry-free test of median difference ($H_0: \operatorname{Median}(d_i) = 0$), the **Wilcoxon Signed-Rank Test** as a complementary location-shift test of the Hodges–Lehmann pseudo-median ($H_0: \theta_{\text{HL}} = 0$) after inspecting symmetry diagnostics, and the **Paired Student's $t$-Test** as a secondary parametric mean-difference reference, with **Holm–Bonferroni FWER ($\alpha = 0.05$)** and **speaker-stratified 10,000-resample bootstrap 95% confidence intervals** (`seed = 42`):

| Comparison ($N = 824$) | Counts: $N^+ / N^- / N^0$ | Median Shift (Stratified Boot 95% CI) | Hodges–Lehmann $\hat{\theta}_{\text{HL}}$ (Wilcoxon 95% CI) | Mean Shift $\pm$ SD (Stratified Boot 95% CI) | Paired Sign Test $p_{\text{Holm}}$ | Wilcoxon Signed-Rank $Z$ ($p_{\text{Holm}}$) | Strict Win Rate ($r_{\text{rb}}$ / Cohen's $d_z$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary: PESQ (FT vs. Zero-Shot)** | `748 / 76 / 0` | **`+0.3977`** (`[+0.3736, +0.4259]`) | **`+0.4036`** (`[+0.3822, +0.4248]`) | **`+0.3977 ± 0.3541`** (`[+0.3735, +0.4214]`) | **`2.4242e-139`** | `22.1238` (**`3.7321e-108`**) | **`90.78%`** (`0.8897` / `1.1231`) |
| **Primary: STOI (FT vs. Zero-Shot)** | `669 / 145 / 10` | **`+0.00845`** (`[+0.0066, +0.0101]`) | **`+0.01360`** (`[+0.0117, +0.0155]`) | **`+0.01795 ± 0.02779`** (`[+0.0161, +0.0198]`) | **`3.7209e-81`** | `19.0113` (**`1.3748e-80`**) | **`81.19%`** (`0.7692` / `0.6458`) |
| **Secondary: PESQ (FT vs. Noisy)** | `810 / 14 / 0` | `+0.9545` (`[+0.9157, +0.9865]`) | `+0.9465` (`[+0.9142, +0.9789]`) | `+0.9460 ± 0.4616` (`[+0.9155, +0.9768]`) | `2.4845e-218` | `24.7970` (`1.9325e-135`) | `98.30%` (`0.9972` / `2.0494`) |
| **Secondary: STOI (FT vs. Noisy)** | `730 / 84 / 10` | `+0.01630` (`[+0.0144, +0.0192]`) | `+0.02215` (`[+0.0200, +0.0244]`) | `+0.02690 ± 0.03286` (`[+0.0248, +0.0292]`) | `2.2942e-129` | `22.1941` (`3.9185e-109`) | `88.59%` (`0.8980` / `0.8186`) |
| **Secondary: PESQ (FT vs. Spectral Gating)** | `818 / 6 / 0` | `+1.5446` (`[+1.4827, +1.5905]`) | `+1.5104` (`[+1.4670, +1.5538]`) | `+1.4948 ± 0.6041` (`[+1.4548, +1.5350]`) | `2.3064e-233` | `24.8603` (`7.9962e-136`) | `99.27%` (`0.9997` / `2.4746`) |
| **Secondary: STOI (FT vs. Spectral Gating)** | `819 / 5 / 0` | `+0.12370` (`[+0.1198, +0.1262]`) | `+0.12350` (`[+0.1205, +0.1265]`) | `+0.12453 ± 0.04717` (`[+0.1213, +0.1277]`) | `2.2501e-235` | `24.8589` (`7.9962e-136`) | `99.39%` (`0.9997` / `2.6400`) |

- **Exploratory Operational Categories (FT vs. Zero-Shot, $N = 824$):**
  - **Wideband PESQ ($\tau_{\text{mod}} = 0.10, \tau_{\text{subst}} = 0.35$ MOS-LQO):** **472 (`57.28%`)** Substantial Improvement ($\ge +0.35$), **227 (`27.55%`)** Moderate Improvement ($[+0.10, +0.35)$), **78 (`9.47%`)** Operational Equivalence ($|\Delta\text{PESQ}| < 0.10$), **25 (`3.03%`)** Moderate Degradation, and **22 (`2.67%`)** Substantial Degradation ($\le -0.35$).
  - **STOI ($\tau_{\text{mod}} = 0.01, \tau_{\text{subst}} = 0.03$):** **204 (`24.76%`)** Substantial Improvement ($\ge +0.03$), **184 (`22.33%`)** Moderate Improvement ($[+0.01, +0.03)$), **396 (`48.06%`)** Operational Equivalence ($|\Delta\text{STOI}| < 0.01$), **32 (`3.88%`)** Moderate Degradation, and **8 (`0.97%`)** Substantial Degradation ($\le -0.03$).

---

## Methodological & Experimental Limitations

1. **Two-Speaker Test Benchmark Clustering (`p232` and `p257`):** The official Edinburgh VoiceBank-DEMAND test split contains 824 utterances recorded by only 2 unseen speakers (`p232`: 393 utterances; `p257`: 431 utterances). While separate speaker tests and speaker-stratified bootstrapping ($B = 10{,}000$) confirm consistent improvements for both individuals, resampling utterances within 2 fixed speaker clusters cannot estimate between-speaker random-effects variance across a general human population. Therefore, pooled statistical inferences are scoped strictly to this 824-utterance benchmark.
2. **STOI Ceiling Compression at High Input SNRs (`12.5 dB` and `17.5 dB`):** Because baseline zero-shot STOI already averages `0.9444` at `12.5 dB` and `0.9594` at `17.5 dB` (medians `0.9569` and `0.9787`), positive STOI gains are bounded above by the mathematical ceiling of `1.0000` (Taal et al., 2011), inducing positive skewness ($g_1 = +1.4740$) in paired STOI differences. Consequently, while STOI improvement at `17.5 dB` is statistically significant ($p_{\text{Holm}} = 1.21 \times 10^{-9}$), its practical magnitude (`+0.0023` median, `+0.0081` mean) lies largely within the operational equivalence band, whereas larger practical intelligibility gains occur at `2.5 dB` (`+0.0283` mean) and `7.5 dB` (`+0.0214` mean).
3. **Stationary Spectral Gating Limitation on Non-Stationary DEMAND Noises:** Without an isolated noise-only reference segment (`y_noise=None`), `noisereduce` estimates stationary noise thresholds across the entire active speech utterance (`n_std_thresh_stationary=1.5, prop_decrease=1.0`), attenuating speech formants and introducing musical-noise artifacts that reduce PESQ (`1.4232`) and STOI (`0.8234`) below unprocessed noisy audio.
4. **Objective Perceptual Surrogates:** ITU-T P.862.2 Wideband PESQ and STOI are validated objective computational surrogates rather than direct human subjective MOS or word-recognition listening panel evaluations.

---

## Dataset Attribution & References

- **Dataset Citation & License:**
  - Valentini-Botinhao, C., Wang, X., Takaki, S., & Yamagishi, J. (2016). *Investigating RNN-based speech enhancement methods for noise-robust Text-to-Speech.* In *Proc. 9th ISCA Speech Synthesis Workshop (SSW9)*, pp. 146–152.
  - Valentini-Botinhao, C. (2017). *Noisy speech database for training speech enhancement algorithms and TTS models.* University of Edinburgh, School of Informatics, Centre for Speech Technology Research (CSTR). Edinburgh DataShare, [https://doi.org/10.7488/ds/2117](https://doi.org/10.7488/ds/2117) (Handle: `10283/2791`). Licensed under **Creative Commons Attribution 4.0 International (`CC BY 4.0`)** — full verbatim license text is preserved in [bridge/verified_license_text.txt](bridge/verified_license_text.txt).
- **Pretrained Architecture Reference:**
  - Defossez, A., Synnaeve, G., & Adi, Y. (2020). *Real Time Speech Enhancement in the Waveform Domain.* In *Proc. Interspeech 2020*, pp. 3291–3295 (`facebookresearch/denoiser`, `dns48` pretrained model).
- **Metric References:**
  - ITU-T Recommendation P.862.2 (2007). *Wideband extension to Recommendation P.862 for the assessment of wideband telephone networks and speech codecs.*
  - Hu, Y., & Loizou, P. C. (2008). *Evaluation of objective quality measures for speech enhancement.* *IEEE Transactions on Audio, Speech, and Language Processing*, 16(1), 229–238.
  - Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2011). *An algorithm for intelligibility prediction of time–frequency weighted noisy speech.* *IEEE Transactions on Audio, Speech, and Language Processing*, 19(7), 2125–2136.

---

## Directory Structure

```text
Speech-Enhancement/
├── .gitignore
├── README.md
├── requirements.txt
├── Speech_Enhancement_Offline_Audio_Demo.zip
├── Speech-Ench-Answers/
│   ├── README.md
│   ├── COMPLETE_PROJECT_REPORTS_AI_CONTEXT.md
│   ├── reports_index.json
│   ├── 01_Speech_Enhancement_Technical_Research_Report.md
│   ├── ...
│   ├── 17_Phase_5A_Final_Repository_Verification_Report.md
│   └── 18_Phase_5B_Offline_Audio_Demo_Report.md
├── audio_demo/
│   ├── README.md
│   ├── audio_integrity_manifest.json
│   ├── index.html
│   └── audio/
│       ├── p232_012/
│       │   ├── clean.wav
│       │   ├── dns48_finetuned.wav
│       │   ├── dns48_zeroshot.wav
│       │   ├── noisy.wav
│       │   └── spectral_gating.wav
│       ├── p232_023/
│       │   └── ...
│       ├── p257_008/
│       │   └── ...
│       └── p257_014/
│           └── ...
├── bridge/
│   ├── step2_audit_evidence.json
│   ├── step2_fetch_docs.py
│   ├── step2_parse_metadata.py
│   ├── step2_run_real_pair.py
│   └── verified_license_text.txt
├── configs/
│   └── config.yaml
├── manifests/
│   ├── train_manifest.csv
│   ├── train_manifest.json
│   ├── val_manifest.csv
│   ├── val_manifest.json
│   ├── test_manifest.csv
│   └── test_manifest.json
├── notebooks/
│   └── Colab_Agent_Bridge.ipynb
├── results/
│   ├── audio_examples/
│   │   ├── p232_012/
│   │   ├── p232_023/
│   │   ├── p257_008/
│   │   └── p257_014/
│   ├── statistical_analysis/
│   │   ├── exploratory_subgroup_tests.csv
│   │   ├── operational_threshold_categories.csv
│   │   ├── operational_threshold_sensitivity.csv
│   │   ├── primary_and_secondary_tests.csv
│   │   ├── statistical_analysis_full_report.json
│   │   ├── statistical_diagnostics_and_effect_sizes.svg
│   │   └── symmetry_and_normality_diagnostics.csv
│   ├── dataset_validation_report.json
│   ├── download_report.json
│   ├── final_test_comparison.csv
│   ├── final_test_comparison.json
│   ├── final_test_comparison.png
│   ├── phase3c_checkpoint_audit_evidence.json
│   ├── preflight_report.json
│   ├── resample_report.json
│   ├── storage_check.json
│   ├── test_baseline_per_utterance.csv
│   ├── test_baseline_summary.csv
│   ├── test_baseline_summary.json
│   ├── test_breakdown_by_noise.csv
│   ├── test_breakdown_by_snr.csv
│   ├── test_breakdown_by_speaker.csv
│   ├── test_finetuned_per_utterance.csv
│   ├── test_finetuned_summary.json
│   ├── training_curves.png
│   ├── training_history.csv
│   ├── training_summary.json
│   ├── val_baseline_per_utterance.csv
│   ├── val_baseline_summary.csv
│   └── val_baseline_summary.json
├── scripts/
│   ├── 01_download_dataset.py
│   ├── 02_resample_16k.py
│   ├── 03_create_manifests_and_validate.py
│   ├── 04_smoke_test.py
│   ├── 05_evaluate_baselines.py
│   ├── 06_finetune_demucs.py
│   ├── 07_evaluate_and_report.py
│   ├── check_storage_and_remote_sizes.py
│   └── test_phase3a_preliminary.py
└── src/
    ├── __init__.py
    ├── baselines/
    │   ├── __init__.py
    │   └── spectral_gating.py
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── download.py
    │   ├── preprocess.py
    │   └── splits.py
    ├── evaluation/
    │   ├── __init__.py
    │   └── evaluator.py
    ├── models/
    │   ├── __init__.py
    │   └── demucs_loader.py
    ├── training/
    │   ├── __init__.py
    │   ├── stft_loss.py
    │   └── trainer.py
    └── utils/
        ├── __init__.py
        ├── audio_io.py
        └── seed.py
```

---

## Reproducible Pipeline Execution (Google Colab GPU / Linux / Windows)

All pipeline paths are centrally configured in [configs/config.yaml](configs/config.yaml) and can be overridden via CLI flags (`--config`, `--data-dir`, `--checkpoints-dir`, `--results-dir`, `--output-dir`). For automated Google Colab execution, open [notebooks/Colab_Agent_Bridge.ipynb](notebooks/Colab_Agent_Bridge.ipynb).

### 1. Environment Installation
Install `facebookresearch/denoiser` with `--no-deps` (to bypass its legacy `hydra-core<1.0` pin on modern Python 3.10–3.13 / PyTorch 2.x environments), followed by `requirements.txt`:
```bash
pip install --no-deps git+https://github.com/facebookresearch/denoiser.git
pip install -r requirements.txt
```

### 2. Download Official 48 kHz Archives
Downloads all 8 official Edinburgh DataShare bitstreams (~5.26 GB) to persistent storage and verifies byte counts, SHA-256 hashes, and ZIP CRC-32 integrity:
```bash
python scripts/01_download_dataset.py
```

### 3. Resample to 16 kHz Mono & Package Persistent Tar Archive
Extracts raw 48 kHz archives, validates mono channel properties, parallel-resamples all 24,792 WAV files to 16 kHz 16-bit PCM mono (`PCM_16`), packages `edinburgh_16k.tar` (2.21 GB), and cleans up temporary extraction files:
```bash
python scripts/02_resample_16k.py
```

### 4. Generate Speaker-Disjoint Manifests & Validate Dataset
Parses official condition logfiles, verifies every clean/noisy utterance pair (existence, frame alignment, finiteness), enforces speaker isolation across Train (26 speakers), Validation (`p226`, `p287`), and Test (`p232`, `p257`), and writes JSON/CSV manifests:
```bash
python scripts/03_create_manifests_and_validate.py
```

### 5. Run Smoke Tests & Preliminary Integration Suite
Runs the synthetic and real-audio pipeline smoke test (`04_smoke_test.py`) and the Phase 3A integration test suite (`test_phase3a_preliminary.py`):
```bash
python scripts/04_smoke_test.py
python scripts/test_phase3a_preliminary.py
```

### 6. Evaluate Phase 3A Baselines
Evaluates Noisy audio, Stationary Spectral Gating, and Pretrained Demucs `dns48` Zero-Shot across all 770 validation and 824 test utterances:
```bash
python scripts/05_evaluate_baselines.py --split all
```

### 7. Fine-Tune Demucs `dns48` (Preflight & 15-Epoch Training)
Run the controlled preflight check and GPU profiler first, then execute 15-epoch fine-tuning:
```bash
python scripts/06_finetune_demucs.py --preflight-only
python scripts/06_finetune_demucs.py
```

### 8. Official Test-Set Evaluation & 4-Way Comparative Report
Run the pre-evaluation preflight check first, then evaluate the selected validation checkpoint (`best_val_pesq.th`, Epoch 14) on all 824 official test utterances:
```bash
python scripts/07_evaluate_and_report.py --preflight-only
python scripts/07_evaluate_and_report.py
```
