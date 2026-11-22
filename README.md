# Speech-Enhancement

Academic speech enhancement system reducing background noise while preserving perceptual speech quality and intelligibility.

## Project Overview

- **Dataset:** University of Edinburgh Noisy Speech Database ([DOI: 10.7488/ds/2117](https://doi.org/10.7488/ds/2117), Handle: `10283/2791`, 28-speaker VoiceBank-DEMAND), resampled to **16 kHz 16-bit PCM mono**.
- **Main Pretrained Model:** Facebook Research Causal Demucs (`dns48` checkpoint, `hidden=48`, 16 kHz waveform U-Net + LSTM).
- **Traditional Baseline:** Stationary Spectral Gating (`noisereduce`).
- **Loss Function:** Time-domain $L_1$ loss + Multi-Resolution STFT loss (`stft_sc_factor = 0.1`, `stft_mag_factor = 0.1`).
- **Evaluation Metrics:** Wideband PESQ (`16 kHz`, `'wb'`) and STOI (`16 kHz`, `extended=False`) across all 824 official test utterances and 770 validation utterances.

---

## Dataset Splits & Verification Statistics

Full dataset integrity verification completed with **0 anomalies** across all 12,396 utterances.

| Split | Speakers (Count) | Utterances | Duration | Noise Types | SNR (dB) Levels |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 26 speakers (`p227`, `p228`, `p230`, `p231`, `p233`, `p236`, `p239`, `p243`, `p244`, `p250`, `p254`, `p256`, `p258`, `p259`, `p267`, `p268`, `p269`, `p270`, `p273`, `p274`, `p276`, `p277`, `p278`, `p279`, `p282`, `p286`) | 10,802 (12,272 chunks @ 4.0s) | 8.758 hours | 10 DEMAND noises (`babble`, `cafeteria`, `car`, `kitchen`, `meeting`, `metro`, `restaurant`, `ssn`, `station`, `traffic`) | 0.0, 5.0, 10.0, 15.0 |
| **Validation** | 2 speakers (`p226`, `p287`) | 770 | 0.633 hours (38.0 min) | 10 DEMAND noises (same as train) | 0.0, 5.0, 10.0, 15.0 |
| **Test** | 2 speakers (`p232`, `p257`) | 824 | 0.576 hours (34.5 min) | 5 unseen DEMAND noises (`bus`, `cafe`, `living`, `office`, `psquare`) | 2.5, 7.5, 12.5, 17.5 |
| **Total** | **30 unique speakers** | **12,396 pairs** | **9.967 hours** | **15 distinct noise types** | **Zero anomalies** |

> [!NOTE]
> **Strict Speaker Isolation:** Test speakers (`p232` and `p257`) have **zero overlap** with the train or validation sets (`overlap: []`). Validation speakers (`p226` and `p287`) are held out completely from model training.

---

## Baseline Benchmark Results (Phase 3A)

Evaluation conducted on all 824 official test utterances and all 770 validation utterances using Wideband PESQ and STOI at 16 kHz (**0 failed evaluations across all 1,594 audio pairs**).

### 1. Official Test Set (824 Utterances, Unseen Noise Types & Speakers)

| Condition | PESQ (Mean ± Std) | PESQ Median | STOI (Mean ± Std) | STOI Median | $\Delta$ PESQ vs Noisy | $\Delta$ STOI vs Noisy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Noisy Audio (Unprocessed)** | 1.9719 ± 0.7532 | 1.7986 | 0.9211 ± 0.0709 | 0.9403 | — | — |
| **Stationary Spectral Gating** | 1.4232 ± 0.2848 | 1.3502 | 0.8234 ± 0.0655 | 0.8320 | -0.5487 | -0.0977 |
| **Pretrained Demucs `dns48` (Zero-Shot)** | **2.5203 ± 0.6258** | **2.5263** | **0.9300 ± 0.0638** | **0.9476** | **+0.5484** | **+0.0089** |

#### Test Set Breakdown by SNR Level
| SNR (dB) | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs `dns48` PESQ | Noisy STOI | Demucs `dns48` STOI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2.5 dB** | 207 | 1.4232 ± 0.4251 | 1.2810 ± 0.2183 | **1.9860 ± 0.4584** (+0.56) | 0.8683 ± 0.0864 | **0.8907 ± 0.0748** (+0.022) |
| **7.5 dB** | 206 | 1.7651 ± 0.5921 | 1.3968 ± 0.2501 | **2.3774 ± 0.5122** (+0.61) | 0.9188 ± 0.0590 | **0.9260 ± 0.0573** (+0.007) |
| **12.5 dB** | 207 | 2.1041 ± 0.6947 | 1.4446 ± 0.2574 | **2.6599 ± 0.5211** (+0.56) | 0.9403 ± 0.0501 | **0.9444 ± 0.0500** (+0.004) |
| **17.5 dB** | 204 | 2.6035 ± 0.7063 | 1.5722 ± 0.3240 | **3.0651 ± 0.4485** (+0.46) | 0.9574 ± 0.0464 | **0.9594 ± 0.0478** (+0.002) |

#### Test Set Breakdown by Unseen Noise Type
| Noise Type | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs `dns48` PESQ | Noisy STOI | Demucs `dns48` STOI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **bus** | 164 | 2.4845 ± 0.7773 | 1.5220 ± 0.3188 | **2.8184 ± 0.5711** (+0.33) | 0.9519 ± 0.0472 | 0.9488 ± 0.0496 |
| **cafe** | 164 | 1.4919 ± 0.4302 | 1.2904 ± 0.1989 | **2.1181 ± 0.5447** (+0.63) | 0.8768 ± 0.0861 | **0.9032 ± 0.0721** (+0.026) |
| **living** | 166 | 1.6178 ± 0.4898 | 1.3628 ± 0.2245 | **2.3150 ± 0.5432** (+0.70) | 0.9040 ± 0.0626 | **0.9218 ± 0.0605** (+0.018) |
| **office** | 164 | 2.5361 ± 0.7057 | 1.5682 ± 0.3317 | **2.9006 ± 0.5518** (+0.36) | 0.9583 ± 0.0559 | 0.9498 ± 0.0622 |
| **psquare** | 166 | 1.7367 ± 0.5673 | 1.3738 ± 0.2237 | **2.4527 ± 0.5435** (+0.72) | 0.9145 ± 0.0621 | **0.9266 ± 0.0605** (+0.012) |

---

### 2. Validation Set (770 Utterances, Held-Out Speakers `p226` and `p287`)

| Condition | PESQ (Mean ± Std) | PESQ Median | STOI (Mean ± Std) | STOI Median | $\Delta$ PESQ vs Noisy | $\Delta$ STOI vs Noisy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Noisy Audio (Unprocessed)** | 1.5714 ± 0.5350 | 1.4186 | 0.8126 ± 0.1118 | 0.8286 | — | — |
| **Stationary Spectral Gating** | 1.2744 ± 0.1960 | 1.2239 | 0.7392 ± 0.0990 | 0.7589 | -0.2970 | -0.0734 |
| **Pretrained Demucs `dns48` (Zero-Shot)** | **2.0691 ± 0.6258** | **2.0111** | **0.8523 ± 0.0964** | **0.8691** | **+0.4977** | **+0.0397** |

#### Validation Set Breakdown by SNR Level
| SNR (dB) | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs `dns48` PESQ | Noisy STOI | Demucs `dns48` STOI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0.0 dB** | 193 | 1.2333 ± 0.2571 | 1.1718 ± 0.1434 | **1.6032 ± 0.4670** (+0.37) | 0.7278 ± 0.1276 | **0.7897 ± 0.1166** (+0.062) |
| **5.0 dB** | 193 | 1.3908 ± 0.3736 | 1.2412 ± 0.1609 | **1.8967 ± 0.5383** (+0.51) | 0.7942 ± 0.0896 | **0.8412 ± 0.0815** (+0.047) |
| **10.0 dB** | 191 | 1.6780 ± 0.5368 | 1.2976 ± 0.1711 | **2.2241 ± 0.5329** (+0.55) | 0.8503 ± 0.0825 | **0.8793 ± 0.0719** (+0.029) |
| **15.0 dB** | 193 | 1.9845 ± 0.5677 | 1.3874 ± 0.2299 | **2.5539 ± 0.5174** (+0.57) | 0.8784 ± 0.0745 | **0.8995 ± 0.0688** (+0.021) |

---

## Fine-Tuning Results (Phase 3B — Demucs `dns48`)

Fine-tuning conducted on the active Google Colab NVIDIA L4 GPU runtime across 15 full epochs. Validation evaluated on all 770 full-length validation utterances (held-out speakers `p226` and `p287`).

> [!IMPORTANT]
> **Strict Evaluation Isolation:** In accordance with project protocol, the official test set (824 utterances, speakers `p232` and `p257`) was **strictly withheld** during Phase 3B fine-tuning and validation. Official test set benchmarking is reserved exclusively for Phase 4.

### 1. Training Configuration & Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | Demucs `dns48` | Causal waveform U-Net + LSTM (`hidden=48`, 16 kHz) |
| **Training Dataset** | 10,802 utterances | 12,272 sliding-window chunks (4.0s segment, 2.0s stride) across 26 train speakers |
| **Batch Size** | 16 | 767 batches per epoch (12,272 chunks / 16) |
| **Optimizer** | Adam | $\text{lr}=5 \times 10^{-5}$, $\beta=(0.9, 0.999)$, $\text{weight\_decay}=0.0$ |
| **Gradient Clipping** | 5.0 | Max norm gradient clipping |
| **Loss Function** | Multi-Resolution STFT + $L_1$ | $\mathcal{L}_{\text{total}} = 1.0 \times \mathcal{L}_1 + 0.1 \times \mathcal{L}_{\text{SC}} + 0.1 \times \mathcal{L}_{\text{Mag}}$ |
| **STFT Resolutions** | 3 filter banks | FFT sizes: `[1024, 2048, 512]`, Hops: `[120, 240, 50]`, Windows: `[600, 1200, 240]` |
| **Training Epochs** | 15 full epochs | No early stopping; atomic checkpointing per epoch |
| **Hardware** | NVIDIA L4 GPU (24 GB) | Steady-state: ~270s/epoch, Peak GPU memory: 4,189.64 MB |

### 2. Validation Progression vs. Baselines (770 Held-Out Utterances)

| Model / Condition | Val Loss | PESQ (Mean ± Std) | PESQ Median | STOI (Mean ± Std) | STOI Median | $\Delta$ PESQ vs Noisy | $\Delta$ PESQ vs Zero-Shot |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Noisy Audio (Unprocessed)** | — | 1.5714 ± 0.5350 | 1.4186 | 0.8126 ± 0.1118 | 0.8286 | — | — |
| **Stationary Spectral Gating** | — | 1.2744 ± 0.1960 | 1.2239 | 0.7392 ± 0.0990 | 0.7589 | -0.2970 | -0.7947 |
| **Pretrained Demucs `dns48` (Epoch 00)** | 0.1342 | 2.0691 ± 0.6258 | 2.0111 | 0.8523 ± 0.0964 | 0.8691 | +0.4977 | — |
| **Fine-Tuned Demucs (Best Val Loss, Ep 9)** | **0.0919** | 2.5614 ± 0.6629 | 2.5648 | **0.8844 ± 0.0764** | **0.8989** | +0.9900 | +0.4923 |
| **Fine-Tuned Demucs (Best Val PESQ, Ep 14)** | 0.0920 | **2.5691 ± 0.6668** | **2.5843** | 0.8835 ± 0.0784 | 0.8981 | **+0.9977** | **+0.5000** |
| **Fine-Tuned Demucs (Final Epoch 15)** | 0.0924 | 2.5646 ± 0.6749 | 2.5697 | 0.8831 ± 0.0791 | 0.8966 | +0.9932 | +0.4955 |

### 3. Epoch-by-Epoch Training & Validation Progression

| Epoch | Train Loss | Train $L_1$ | Train SC | Train Mag | Val Loss | Val $L_1$ | Val SC | Val Mag | Val PESQ | Val STOI | Duration | Peak GPU RAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00** | — | — | — | — | 0.1342 | 0.0092 | 0.1801 | 1.0703 | 2.0691 | 0.8523 | 40.5s | 248.9 MB |
| **01** | 0.0758 | 0.0053 | 0.1658 | 0.5389 | 0.0965 | 0.0075 | 0.1255 | 0.7650 | 2.4823 | 0.8759 | 268.8s | 4,189.6 MB |
| **02** | 0.0705 | 0.0050 | 0.1467 | 0.5080 | 0.0950 | 0.0073 | 0.1235 | 0.7530 | 2.5078 | 0.8786 | 269.8s | 4,189.6 MB |
| **03** | 0.0683 | 0.0049 | 0.1377 | 0.4963 | 0.0942 | 0.0073 | 0.1225 | 0.7463 | 2.5295 | 0.8798 | 269.7s | 4,189.6 MB |
| **04** | 0.0668 | 0.0048 | 0.1324 | 0.4871 | 0.0933 | 0.0073 | 0.1210 | 0.7389 | 2.5259 | 0.8818 | 269.7s | 4,189.6 MB |
| **05** | 0.0656 | 0.0048 | 0.1291 | 0.4791 | 0.0930 | 0.0073 | 0.1213 | 0.7355 | 2.5322 | 0.8817 | 270.1s | 4,189.6 MB |
| **06** | 0.0647 | 0.0047 | 0.1270 | 0.4731 | 0.0925 | 0.0073 | 0.1213 | 0.7308 | 2.5523 | 0.8825 | 269.9s | 4,189.6 MB |
| **07** | 0.0639 | 0.0047 | 0.1243 | 0.4676 | 0.0927 | 0.0073 | 0.1221 | 0.7314 | 2.5486 | 0.8823 | 269.9s | 4,189.6 MB |
| **08** | 0.0632 | 0.0047 | 0.1227 | 0.4626 | 0.0921 | 0.0073 | 0.1213 | 0.7273 | 2.5567 | 0.8834 | 269.9s | 4,189.6 MB |
| **09** | 0.0625 | 0.0046 | 0.1213 | 0.4574 | **0.0919** | 0.0072 | 0.1201 | 0.7271 | 2.5614 | **0.8844** | 270.1s | 4,189.6 MB |
| **10** | 0.0618 | 0.0046 | 0.1198 | 0.4522 | 0.0923 | 0.0073 | 0.1217 | 0.7290 | 2.5634 | 0.8832 | 270.0s | 4,189.6 MB |
| **11** | 0.0612 | 0.0046 | 0.1186 | 0.4475 | 0.0924 | 0.0073 | 0.1219 | 0.7284 | 2.5572 | 0.8839 | 269.9s | 4,189.6 MB |
| **12** | 0.0607 | 0.0046 | 0.1174 | 0.4438 | 0.0923 | 0.0073 | 0.1217 | 0.7289 | 2.5556 | 0.8835 | 270.3s | 4,189.6 MB |
| **13** | 0.0601 | 0.0045 | 0.1165 | 0.4388 | 0.0920 | 0.0072 | 0.1213 | 0.7267 | 2.5528 | 0.8839 | 270.1s | 4,189.6 MB |
| **14** | 0.0594 | 0.0045 | 0.1157 | 0.4336 | 0.0920 | 0.0073 | 0.1215 | 0.7263 | **2.5691** | 0.8835 | 270.1s | 4,189.6 MB |
| **15** | **0.0590** | 0.0045 | 0.1148 | 0.4305 | 0.0924 | 0.0073 | 0.1223 | 0.7290 | 2.5646 | 0.8831 | 270.2s | 4,189.6 MB |

> [!TIP]
> The full loss, PESQ, and STOI curves are visually charted in [training_curves.png](file:///c:/Users/yilma/OneDrive/Desktop/Speech-Enhancement/results/training_curves.png). Persistent model checkpoints are stored in `/content/drive/MyDrive/Speech-Enhancement/checkpoints/` (`best_val_pesq.th`, `best_val_loss.th`, and `checkpoint.th`).

---

## Official Test-Set Evaluation & 4-Way Comparative Results (Phase 4)

Comprehensive evaluation conducted across all **824 official Edinburgh VoiceBank-DEMAND test utterances** (unseen speakers `p232` and `p257`, 5 unseen DEMAND noise types, and 4 SNR levels) on the active Google Colab NVIDIA L4 GPU runtime.

The fine-tuned model checkpoint (`best_val_pesq.th`, selected at Epoch 14 via validation PESQ) was evaluated with zero retraining, and compared 1-to-1 against the verified Phase 3A baselines across all 824 paired utterances.

### 1. Overall Performance Comparison (824 Test Utterances)

| Condition / Model | PESQ (Mean ± Std) | PESQ Median | STOI (Mean ± Std) | STOI Median | $\Delta$ PESQ vs Noisy | $\Delta$ PESQ vs Zero-Shot | $\Delta$ STOI vs Noisy | $\Delta$ STOI vs Zero-Shot | Successful / Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Noisy Audio (Unprocessed)** | 1.9719 ± 0.7532 | 1.7986 | 0.9211 ± 0.0709 | 0.9403 | — | -0.5484 (-21.8%) | — | -0.0089 (-1.0%) | 824 / 824 |
| **Stationary Spectral Gating** | 1.4232 ± 0.2848 | 1.3502 | 0.8234 ± 0.0655 | 0.8320 | -0.5487 (-27.8%) | -1.0971 (-43.5%) | -0.0977 (-10.6%) | -0.1066 (-11.5%) | 824 / 824 |
| **Pretrained Demucs `dns48` (Zero-Shot)** | 2.5203 ± 0.6258 | 2.5263 | 0.9300 ± 0.0638 | 0.9476 | +0.5484 (+27.8%) | — | +0.0089 (+1.0%) | — | 824 / 824 |
| **Fine-Tuned Demucs `dns48` (Epoch 14)** | **2.9180 ± 0.6483** | **3.0316** | **0.9480 ± 0.0507** | **0.9645** | **+0.9461 (+48.0%)** | **+0.3977 (+15.8%)** | **+0.0269 (+2.9%)** | **+0.0180 (+1.9%)** | **824 / 824** |

### 2. Performance Breakdown by Test Input SNR Level

| SNR Level | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs `dns48` Zero-Shot PESQ | Fine-Tuned Demucs PESQ | $\Delta$PESQ vs Zero-Shot | Noisy STOI | Demucs `dns48` Zero-Shot STOI | Fine-Tuned Demucs STOI | $\Delta$STOI vs Zero-Shot |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2.5 dB** | 207 | 1.4232 ± 0.4251 | 1.2810 ± 0.2183 | 1.9860 ± 0.4584 | **2.4203 ± 0.6152** | **+0.4343** | 0.8683 ± 0.0864 | 0.8907 ± 0.0748 | **0.9190 ± 0.0604** | **+0.0283** |
| **7.5 dB** | 206 | 1.7651 ± 0.5921 | 1.3968 ± 0.2501 | 2.3774 ± 0.5122 | **2.8218 ± 0.5718** | **+0.4444** | 0.9188 ± 0.0590 | 0.9260 ± 0.0573 | **0.9474 ± 0.0444** | **+0.0214** |
| **12.5 dB** | 207 | 2.1041 ± 0.6947 | 1.4446 ± 0.2574 | 2.6599 ± 0.5211 | **3.0631 ± 0.5385** | **+0.4032** | 0.9403 ± 0.0501 | 0.9444 ± 0.0500 | **0.9582 ± 0.0398** | **+0.0138** |
| **17.5 dB** | 204 | 2.6035 ± 0.7063 | 1.5722 ± 0.3240 | 3.0651 ± 0.4485 | **3.3728 ± 0.4476** | **+0.3078** | 0.9574 ± 0.0464 | 0.9594 ± 0.0478 | **0.9674 ± 0.0416** | **+0.0081** |

### 3. Performance Breakdown by Unseen DEMAND Noise Type

| Noise Type | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs `dns48` Zero-Shot PESQ | Fine-Tuned Demucs PESQ | $\Delta$PESQ vs Zero-Shot | $\Delta$PESQ vs Noisy | Fine-Tuned Demucs STOI | $\Delta$STOI vs Zero-Shot |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **office** | 164 | 2.5361 ± 0.7057 | 1.5682 ± 0.3317 | 2.9006 ± 0.5518 | **3.4342 ± 0.4769** | **+0.5337** | **+0.8982** | **0.9667 ± 0.0464** | **+0.0169** |
| **bus** | 164 | 2.4845 ± 0.7773 | 1.5220 ± 0.3188 | 2.8184 ± 0.5711 | **3.2107 ± 0.4532** | **+0.3922** | **+0.7262** | **0.9635 ± 0.0405** | **+0.0146** |
| **psquare** | 166 | 1.7367 ± 0.5673 | 1.3738 ± 0.2237 | 2.4527 ± 0.5435 | **2.8194 ± 0.5205** | **+0.3666** | **+1.0827** | **0.9430 ± 0.0490** | **+0.0165** |
| **cafe** | 164 | 1.4919 ± 0.4302 | 1.2904 ± 0.1989 | 2.1181 ± 0.5447 | **2.4668 ± 0.6115** | **+0.3487** | **+0.9749** | **0.9256 ± 0.0586** | **+0.0224** |
| **living** | 166 | 1.6178 ± 0.4898 | 1.3628 ± 0.2245 | 2.3150 ± 0.5432 | **2.6632 ± 0.6283** | **+0.3482** | **+1.0454** | **0.9411 ± 0.0455** | **+0.0193** |

### 4. Performance Breakdown by Unseen Test Speaker

| Speaker | Gender | Utterances | Noisy PESQ | Spectral Gating PESQ | Demucs `dns48` Zero-Shot PESQ | Fine-Tuned Demucs PESQ | $\Delta$PESQ vs Zero-Shot | Fine-Tuned Demucs STOI | $\Delta$STOI vs Zero-Shot |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **p232** | Male | 393 | 2.1687 ± 0.7147 | 1.3991 ± 0.2847 | 2.6535 ± 0.5909 | **3.0351 ± 0.5971** | **+0.3817** | **0.9504 ± 0.0504** | **+0.0104** |
| **p257** | Female | 431 | 1.7925 ± 0.7425 | 1.4451 ± 0.2831 | 2.3989 ± 0.6319 | **2.8112 ± 0.6742** | **+0.4123** | **0.9457 ± 0.0508** | **+0.0248** |

### 5. Utterance-Level Diagnostic Analysis

- **Improvement Frequency over Zero-Shot:** **748 out of 824 utterances (90.78%)** achieved higher Wideband PESQ after fine-tuning.
- **Improvement Frequency over Noisy:** **810 out of 824 utterances (98.30%)** showed significant speech enhancement over unprocessed noisy speech.
- **Intelligibility Preservation:** Only 78 utterances (9.47%) experienced a minor drop in STOI while improving PESQ vs Noisy.
- **Single-Pair Smoke Test Utterance (`p232_001.wav`):** Evaluated normally as part of the 824 test set (Noise: `bus`, SNR: `17.5 dB`). Scores: Noisy PESQ = `2.9238` $\to$ Spectral Gating = `1.6644` $\to$ Demucs Zero-Shot = `3.0293` $\to$ **Fine-Tuned Demucs = 3.1112** (STOI = `0.9085`).

### 6. Representative Audio Demonstration Examples

Four representative audio examples covering all 4 SNR levels (`2.5`, `7.5`, `12.5`, `17.5` dB), diverse unseen noises, and both speakers were exported as complete 5-audio comparison sets (`clean.wav`, `noisy.wav`, `spectral_gating.wav`, `dns48_zeroshot.wav`, `dns48_finetuned.wav`) in [audio_examples/](file:///c:/Users/yilma/OneDrive/Desktop/Speech-Enhancement/results/audio_examples/):

| Utterance ID | Speaker | Noise Type | Input SNR | Duration | Noisy PESQ / STOI | Zero-Shot PESQ / STOI | Fine-Tuned PESQ / STOI | Improvement vs Zero-Shot | Audio Directory |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **p232_023** | `p232` (M) | `psquare` | 2.5 dB | 9.77s | 1.3108 / 0.7719 | 2.2626 / 0.8878 | **2.5008 / 0.9026** | **+0.2382 PESQ, +0.0148 STOI** | [audio files](file:///c:/Users/yilma/OneDrive/Desktop/Speech-Enhancement/results/audio_examples/p232_023/) |
| **p257_008** | `p257` (F) | `cafe` | 7.5 dB | 5.58s | 1.2496 / 0.9052 | 1.9051 / 0.9325 | **2.2039 / 0.9531** | **+0.2988 PESQ, +0.0206 STOI** | [audio files](file:///c:/Users/yilma/OneDrive/Desktop/Speech-Enhancement/results/audio_examples/p257_008/) |
| **p232_012** | `p232` (M) | `living` | 12.5 dB | 4.34s | 2.0530 / 0.9654 | 2.7081 / 0.9648 | **3.1346 / 0.9739** | **+0.4265 PESQ, +0.0091 STOI** | [audio files](file:///c:/Users/yilma/OneDrive/Desktop/Speech-Enhancement/results/audio_examples/p232_012/) |
| **p257_014** | `p257` (F) | `office` | 17.5 dB | 3.69s | 3.3299 / 0.9944 | 3.4156 / 0.9950 | **3.7012 / 0.9961** | **+0.2856 PESQ, +0.0011 STOI** | [audio files](file:///c:/Users/yilma/OneDrive/Desktop/Speech-Enhancement/results/audio_examples/p257_014/) |

> [!NOTE]
> The full publication-grade 4-panel figure comparing all conditions across SNRs, noises, and per-utterance distributions is available in [final_test_comparison.png](file:///c:/Users/yilma/OneDrive/Desktop/Speech-Enhancement/results/final_test_comparison.png). Complete tabular data is stored in [final_test_comparison.csv](file:///c:/Users/yilma/OneDrive/Desktop/Speech-Enhancement/results/final_test_comparison.csv) and [final_test_comparison.json](file:///c:/Users/yilma/OneDrive/Desktop/Speech-Enhancement/results/final_test_comparison.json).

---

## Directory Structure

```text
Speech-Enhancement/
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── manifests/
│   ├── train_manifest.csv
│   ├── train_manifest.json
│   ├── val_manifest.csv
│   ├── val_manifest.json
│   ├── test_manifest.csv
│   └── test_manifest.json
├── results/
│   ├── audio_examples/
│   │   ├── p232_012/
│   │   ├── p232_023/
│   │   ├── p257_008/
│   │   └── p257_014/
│   ├── dataset_validation_report.json
│   ├── download_report.json
│   ├── final_test_comparison.csv
│   ├── final_test_comparison.json
│   ├── final_test_comparison.png
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

## Reproducible Pipeline Execution (Google Colab L4 GPU)

### 1. Environment Installation
```bash
pip install --no-deps git+https://github.com/facebookresearch/denoiser.git
pip install -r requirements.txt
```

### 2. Download 48 kHz Archives
Downloads official Edinburgh archives (~5.26 GB) directly to persistent Google Drive storage and verifies SHA-256 and zip integrity:
```bash
python scripts/01_download_dataset.py
```

### 3. Resample to 16 kHz Mono & Create Persistent Tar Archive
Extracts raw archives, validates 48 kHz mono properties, parallel-resamples all 24,792 audio files to 16 kHz 16-bit PCM mono, packages persistent `edinburgh_16k.tar` (2.21 GB) on Google Drive, and removes temporary extracted files:
```bash
python scripts/02_resample_16k.py
```

### 4. Create Manifests & Run Full Validation
Parses official condition logfiles, validates every single utterance pair (audio existence, sample alignment, no NaNs/Infs), enforces strict speaker isolation, and outputs manifests:
```bash
python scripts/03_create_manifests_and_validate.py
```

### 5. Run Preliminary Tests
Executes unit tests verifying dataset loading, Colab runtime auto-restoration from tar, spectral gating edge cases, Demucs inference, and metric calculations:
```bash
python scripts/test_phase3a_preliminary.py
```

### 6. Run Baseline Evaluation
Evaluates Noisy audio, Stationary Spectral Gating, and Pretrained Demucs `dns48` zero-shot across all 770 validation and 824 test utterances using multi-core parallel processing:
```bash
python scripts/05_evaluate_baselines.py --split all
```

### 7. Demucs `dns48` Fine-Tuning (Preflight & Full Training)
First run preflight verification and profiling:
```bash
python scripts/06_finetune_demucs.py --preflight-only
```
Then execute full 15-epoch fine-tuning with multi-resolution STFT + $L_1$ loss and validation metric tracking:
```bash
python scripts/06_finetune_demucs.py
```

### 8. Official Test-Set Evaluation & Unified Comparative Analysis
First run controlled pre-evaluation preflight checks:
```bash
python scripts/07_evaluate_and_report.py --preflight-only
```
Then execute full 824-utterance evaluation of the fine-tuned model checkpoint (`best_val_pesq.th`), compile unified 4-way comparative metrics, generate publication plots, and export representative audio demonstration sets:
```bash
python scripts/07_evaluate_and_report.py
```
