> **Document ID:** `01` | **Phase:** Phase 1A — Technical Research | **Role:** Research / Architecture  
> **Source Word Document:** `Speech_Enhancement_Technical_Research_Report.docx` | **Repository Path:** `Speech-Ench-Answers/01_Speech_Enhancement_Technical_Research_Report.md`

---
# Technical Research Report & Implementation Strategy: Speech-Enhancement

**Project Objective:** Develop a speech enhancement system that reduces background noise while preserving speech quality.

**Assignment Scope:** Dataset: NoisySpeech  |  Methods: DNN, U-Net, Spectral Gating  |  Approach: Fine-tune a pretrained model (Demucs / RNNoise)  |  Metrics: PESQ & STOI  |  Stack: Python, PyTorch, Google Colab GPU, Google Drive, Antigravity 2.0

## PHASE 1: DATASET RESEARCH ("NoisySpeech" Candidates)

In academic speech enhancement literature and coursework, **"NoisySpeech"** is not a unique proper noun for a single registry; rather, it is an umbrella shorthand referring to one of **four primary datasets** (either by literal title or by exact folder naming conventions). Below is a rigorous investigation of all four credible candidates against the 8 evaluation criteria.

### Candidate 1: University of Edinburgh Noisy Speech Database (VoiceBank-DEMAND)

1. **Exact Official Name:** *Noisy speech database for training speech enhancement algorithms and TTS models* (commonly cited in papers as **VoiceBank-DEMAND** or **Valentini-Botinhao**).
2. **Publisher & Official Source:** University of Edinburgh, Centre for Speech Technology Research (CSTR) — Published by Cassia Valentini-Botinhao et al. on **Edinburgh DataShare** (https://datashare.ed.ac.uk/handle/10283/2791, DOI: `10.7488/ds/2117`). Also available in a 56-speaker version (DOI: `10.7488/ds/2139`).
3. **Documentation & Download Availability:** Fully documented (accompanied by ICASSP 2016 / SSW 2016 papers and README/log files detailing speaker IDs, noise types, and SNR per utterance). Direct HTTP/ZIP downloads are publicly available without registration via Edinburgh DataShare: `clean_trainset_28spk_wav.zip` (~2.3 GB), `noisy_trainset_28spk_wav.zip` (~2.3 GB), `clean_testset_wav.zip` (~147 MB), and `noisy_testset_wav.zip` (~147 MB).
4. **Clean and Noisy Speech Pairs: Yes (Pre-mixed & Strictly Aligned).** Contains 11,572 parallel training utterances (28 speakers, 10 noise types at 0, 5, 10, and 15 dB SNR) and 824 parallel test utterances (2 unseen speakers, 5 unseen noise types at 2.5, 7.5, 12.5, and 17.5 dB SNR). Every noisy file (e.g., `p226_001.wav`) has an exact sample-aligned clean reference counterpart.
5. **Audio Format & Sampling Rate:** Uncompressed 16-bit PCM mono `.wav` at 48 kHz natively (must be downsampled to 16 kHz during preprocessing for standard wideband speech enhancement and PESQ).
6. **License & Usage Conditions: Creative Commons Attribution 4.0 International (CC BY 4.0)**. Free for academic, educational, and commercial use with attribution.
7. **Suitability for Training & Testing:** Gold-standard academic benchmark. Because the train and test sets are pre-mixed and fixed, results are 100% reproducible across studies without randomness in the data synthesis pipeline.
8. **PESQ & STOI Support:** Yes. Because clean and noisy pairs are strictly time-aligned, downsampling to 16 kHz directly supports ITU-T P.862.2 Wideband PESQ (`pesq(16000, clean, enhanced, 'wb')`) and `pystoi`.

### Candidate 2: Microsoft Scalable Noisy Speech Dataset (MS-SNSD)

1. **Exact Official Name:** *Microsoft Scalable Noisy Speech Dataset (MS-SNSD)* — specifically notable because running its synthesis script creates literal default folders named `NoisySpeech_training` and `CleanSpeech_training`.
2. **Publisher & Official Source:** Microsoft Corporation (Chandan K. A. Reddy et al., Interspeech 2019). Official GitHub repository: https://github.com/microsoft/MS-SNSD.
3. **Documentation & Download Availability:** Well-documented via the GitHub README and Interspeech 2019 paper ("A Scalable Noisy Speech Dataset and Online Subjective Test Framework"). The repository includes `clean_train`, `clean_test`, `noise_train`, and `noise_test` raw audio folders directly inside the git tree, alongside `noisyspeech_synthesizer.py` and `noisyspeech_synthesizer.cfg`.
4. **Clean and Noisy Speech Pairs: Synthesized on demand (Not pre-mixed by default).** The repo supplies unmixed clean speech and environmental noise tracks. Running `noisyspeech_synthesizer.py` mixes them at user-configured SNR levels (default: 0 to 40 dB) and outputs aligned `.wav` pairs into `NoisySpeech_training/` and `CleanSpeech_training/`.
5. **Audio Format & Sampling Rate:** 16-bit PCM mono `.wav` at 16 kHz natively.
6. **License & Usage Conditions:** Code is licensed under the MIT License; audio files are distributed on an "AS IS" basis under the terms of their underlying third-party sources (LibriSpeech/VCTK/AudioSet subsets) for research use.
7. **Suitability for Training & Testing:** Highly suitable and natively 16 kHz, but because pairs are synthesized dynamically via random SNR/noise selection in `noisyspeech_synthesizer.py`, two students running the script without identical random seeds and config parameters (`total_hours`, `snr_lower`, `snr_upper`) will generate different datasets.
8. **PESQ & STOI Support:** Yes. Synthesized pairs are sample-aligned at 16 kHz, immediately compatible with wideband PESQ and STOI.

### Candidate 3: Microsoft Deep Noise Suppression (DNS) Challenge Datasets

1. **Exact Official Name:** *Microsoft DNS Challenge Dataset (INTERSPEECH 2020 / ICASSP 2021-2023 Deep Noise Suppression Challenge)*.
2. **Publisher & Official Source:** Microsoft Corporation (https://github.com/microsoft/DNS-Challenge, derived from MS-SNSD).
3. **Documentation & Download Availability:** Documented on GitHub; downloaded via shell scripts (`download-dns-challenge-*.sh`) fetching multi-part Azure Blob Storage archives.
4. **Clean and Noisy Speech Pairs:** Unmixed raw corpora for training (requiring synthesis via `noisyspeech_synthesizer_singleprocess.py`), plus a small pre-mixed blind/non-blind test set (`synthetic/no_reverb`).
5. **Audio Format & Sampling Rate:** 16-bit PCM `.wav` at 16 kHz (v1) and 48 kHz (v2+).
6. **License & Usage Conditions:** Code under MIT / CC-BY 4.0 depending on version.
7. **Suitability for Training & Testing:** Poor fit for Google Colab & Google Drive. The full dataset ranges from 80 GB to 500+ GB, exceeding standard free Google Drive quotas (15 GB) and causing severe I/O bottlenecks in Colab.
8. **PESQ & STOI Support:** Yes (on the synthetic test set and synthesized pairs).

### Candidate 4: NOIZEUS (Noisy Speech Corpus)

1. **Exact Official Name:** *NOIZEUS: A noisy speech corpus for evaluation of speech enhancement algorithms*.
2. **Publisher & Official Source:** University of Texas at Dallas — Prof. Philipos C. Loizou (https://ecs.utdallas.edu/loizou/speech/noizeus/).
3. **Documentation & Download Availability:** Documented on the UTDallas academic portal; downloadable as small ZIP archives per noise condition (babble, car, restaurant, street, airport, train, exhibition, station at 0, 5, 10, 15 dB).
4. **Clean and Noisy Speech Pairs:** Yes, pre-mixed and aligned.
5. **Audio Format & Sampling Rate:** 16-bit PCM `.wav` at 8 kHz (narrowband).
6. **License & Usage Conditions:** Academic research use.
7. **Suitability for Training & Testing:** Unsuitable for deep learning training/fine-tuning. Contains only 30 IEEE sentences (~2.5 minutes of audio total). Furthermore, its 8 kHz narrowband sampling rate is incompatible with modern 16 kHz pretrained models (such as Demucs `dns64`, which requires 16 kHz, or RNNoise, which requires 48 kHz).
8. **PESQ & STOI Support:** Supports narrowband PESQ ('nb') and STOI at 8 kHz, but cannot evaluate wideband (16 kHz) speech enhancement without upsampling artifacts.

### Dataset Comparison Matrix & Single-Source Recommendation

| Feature | Candidate 1: Edinburgh Noisy Speech DB (VoiceBank-DEMAND) | Candidate 2: Microsoft MS-SNSD | Candidate 3: Microsoft DNS Challenge | Candidate 4: NOIZEUS |
| :--- | :--- | :--- | :--- | :--- |
| DOI / Source | 10.7488/ds/2117 (Edinburgh DataShare) | github.com/microsoft/MS-SNSD | github.com/microsoft/DNS-Challenge | ecs.utdallas.edu/loizou/speech/noizeus |
| Literal Name Match | Title starts with "Noisy speech database..." | Synthesizer outputs NoisySpeech_training/ | Derived from MS-SNSD | "Noisy speech corpus" |
| Paired Clean/Noisy | Pre-mixed & fixed (11,572 train / 824 test) | Requires running synthesizer script | Requires running synthesizer script | Pre-mixed (only 30 sentences) |
| Sampling Rate | 48 kHz .wav (downsample to 16 kHz) | 16 kHz .wav native | 16 kHz / 48 kHz .wav | 8 kHz .wav (narrowband only) |
| Storage Size | ~4.9 GB (48 kHz) / ~1.6 GB (at 16 kHz) | Configurable (~2–5 GB typical) | 80 GB – 500+ GB | ~50 MB |
| License | CC BY 4.0 | MIT (code) / Research ("AS IS") | MIT / CC BY 4.0 | Academic Research |
| Colab/Drive Fit | Excellent (1.6 GB at 16 kHz fits easily) | Excellent | Poor (exceeds Drive quota) | Too small for DL training |
| PESQ / STOI | Full Support (Wideband 16 kHz + STOI) | Full Support (Wideband 16 kHz + STOI) | Full Support | Narrowband 8 kHz only |

- **Primary Recommendation: Candidate 1 — University of Edinburgh Noisy Speech Database (10.7488/ds/2117, 28-speaker version).** It is literally titled "Noisy speech database for training speech enhancement algorithms and TTS models", requires zero synthetic mixing assumptions, provides a standardized 824-utterance test split with published PESQ/STOI baselines across hundreds of papers (including the official Demucs paper), and takes only ~1.6 GB on Google Drive once resampled to 16 kHz.
- **Explicit Uncertainty Note:** If the course instructor provided a specific link or meant Microsoft MS-SNSD (Candidate 2, whose script generates the literal folder name `NoisySpeech_training`), our pipeline is designed to use strictly ONE dataset source without combining datasets.

## PHASE 2: MODEL RESEARCH & ARCHITECTURAL RECONCILIATION

### 1. How the Listed Methods Relate (Addressing the Assignment Requirements)

The assignment lists Methods (`DNN, U-Net, Spectral Gating`), Model approach (`Fine-tune a pretrained model`), and Suggested model examples (`Demucs and RNNoise`). Crucial Technical Insight: You do not need to build three separate deep learning models from scratch to satisfy DNN, U-Net, and Spectral Gating alongside Demucs / RNNoise. In speech enhancement taxonomy:

1. **Demucs (facebookresearch/denoiser) IS a Deep Neural Network (DNN) built on a U-Net architecture:** Specifically, Demucs (Defossez et al., Interspeech 2020, "Real Time Speech Enhancement in the Waveform Domain") is a 1D Waveform Encoder-Decoder U-Net with skip connections and a bi-directional/causal LSTM bottleneck. Fine-tuning pretrained Demucs simultaneously satisfies DNN, U-Net, Pretrained Model Fine-Tuning, and the suggested example Demucs.
2. **Spectral Gating IS the Traditional DSP Baseline:** Spectral Gating computes the Short-Time Fourier Transform (STFT) of the noisy signal, estimates a frequency-dependent noise threshold (mean + alpha * std across time), applies a soft/hard TF mask (gate), and reconstructs the signal via ISTFT. It requires no GPU training and serves as the ideal classical baseline to benchmark against the fine-tuned U-Net DNN (Demucs).

### 2. Detailed Comparison of the 5 Listed Methods / Models

| Criterion | Spectral Gating (Baseline) | Generic DNN (MLP / Masking) | Standard 2D Spectrogram U-Net | RNNoise (Xiph.org) | Demucs (facebookresearch/denoiser) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Domain & Architecture | Classical DSP (STFT magnitude thresholding & masking) | STFT Log-Power Spectral mapping or Ideal Ratio Mask (IRM) | 2D Conv Encoder-Decoder on STFT Magnitude (reuses noisy phase) | Hybrid DSP + lightweight 3-layer GRU predicting 22 Bark-band gains + pitch | 1D Waveform U-Net (Strided 1D Conv Encoder + Skip Connections + LSTM + Transposed Conv Decoder) |
| Pretrained Checkpoint Availability | N/A (Non-parametric / algorithmic) | Low (No single canonical standalone PyTorch checkpoint) | Moderate (Scattered repo checkpoints; mostly music separation) | Official weights exist in C (rnnoise_data.c); no official pip-installable PyTorch hub | Excellent: Official Meta Research checkpoints (dns48, dns64, master64) via denoiser.pretrained |
| Fine-Tuning Feasibility in PyTorch | N/A (Hyperparameter tuning only: n_std_thresh, smoothing) | High if trained from scratch; poor for standardized fine-tuning | Moderate (Requires custom STFT matching to checkpoint config) | Low / Cumbersome: Requires C compilation (dump_features) to extract 48 kHz .f32 Bark features | Very High: Native PyTorch nn.Module; supports end-to-end waveform loss (L1 + Multi-Res STFT Loss) |
| Dataset Compatibility (16 kHz) | Compatible with any sample rate (16 kHz or 48 kHz) | Compatible (depends on STFT config) | Compatible (depends on STFT config) | Hardcoded to 48 kHz (20 ms frames = 960 samples; 16 kHz breaks C DSP filterbanks) | Natively 16 kHz (dns48, dns64, master64 all operate at sample_rate = 16000 Hz, mono) |
| PyTorch Implementation | noisereduce library (supports PyTorch backend) or ~30 lines of torch.stft | Custom torch.nn implementation required | Custom torch.nn implementation required | Primarily C codebase (gitlab.xiph.org/xiph/rnnoise) with auxiliary scripts | 100% Native PyTorch (pip install denoiser / facebookresearch/denoiser) |
| Colab GPU Requirements | Runs on CPU/GPU in seconds (no training) | Low (1–2 GB VRAM) | Moderate (4–8 GB VRAM) | Low (~1 GB VRAM, bottlenecked by CPU C-feature dumping) | Low-to-Moderate (~4–8 GB VRAM for batch size 4–8 with 4s segments; fits Colab T4 15 GB GPU) |
| Implementation Complexity | Very Low | Moderate | Moderate | High (C/Python cross-compilation & custom DSP pipeline) | Low (Clean API, modular PyTorch architecture) |
| PESQ & STOI Performance | Moderate noise reduction; "musical noise" artifacts limit PESQ/STOI gains | Struggles with phase reconstruction and non-stationary noise | Phase mismatch limits ceiling on PESQ | Good real-time voice communication, but coarse 22-band Bark resolution caps PESQ | State-of-the-art perceptual quality (enhances magnitude & phase in time domain) |

### 3. Model Recommendation

- **Recommended Main Learning-Based Model: Demucs (facebookresearch/denoiser, specifically the dns48 or dns64 pretrained checkpoint).** Crucially, in facebookresearch/denoiser, the master64 checkpoint was trained on both DNS and VoiceBank-DEMAND, whereas dns48 and dns64 were trained strictly on the Microsoft DNS dataset. Therefore, if we use the Edinburgh VoiceBank-DEMAND dataset, fine-tuning dns48 (or dns64) guarantees zero data leakage between pretraining and our target dataset, allowing us to measure genuine domain adaptation and fine-tuning improvements.
- **Recommended Traditional Baseline: Spectral Gating** (implemented cleanly via noisereduce / PyTorch STFT masking on the exact same test split).
- **How Every Assignment Requirement is Satisfied:**
  - **1. Spectral Gating:** Evaluated as Baseline #1 (Traditional DSP method).
  - **2. Pretrained Demucs (Zero-Shot):** Evaluated as Baseline #2 (Out-of-the-box pretrained U-Net DNN prior to domain adaptation).
  - **3. Fine-Tuned Demucs (Main Model):** Satisfies DNN, U-Net, Demucs, and Fine-tuning a pretrained model in PyTorch on Google Colab GPU.

## PHASE 3: COMPLETE IMPLEMENTATION PLAN

### 1. Project Architecture & Folder Structure

We will keep the codebase modular, clean, and reproducible, separating local/Colab source code (`Speech-Enhancement` repo) from large dataset/checkpoint storage on Google Drive:

`Speech-Enhancement/`

`├── configs/`

`│   └── config.yaml               # Central hyperparameters (SR=16000, batch_size, lr, epochs, paths)`

`├── src/`

`│   ├── __init__.py`

`│   ├── dataset.py                # PyTorch Dataset & DataLoader (16 kHz resampling, fixed-length chunking)`

`│   ├── baselines/`

`│   │   └── spectral_gating.py    # Spectral Gating DSP baseline (stationary/non-stationary STFT gate)`

`│   ├── models/`

`│   │   └── demucs_wrapper.py     # Loads facebookresearch/denoiser pretrained dns48/dns64 & layer freezing`

`│   ├── losses.py                 # L1 Waveform Loss + Multi-Resolution STFT Loss`

`│   ├── trainer.py                # Fine-tuning loop, validation, early stopping, Drive checkpointing`

`│   └── evaluate.py               # Wideband PESQ (16 kHz) & STOI batch evaluation + audio sample export`

`├── notebooks/`

`│   └── Colab_Training_Pipeline.ipynb  # End-to-end Google Colab notebook (Drive mount, train, evaluate, plots)`

`├── scripts/`

`│   ├── prepare_data.py           # Downloads/verifies dataset & resamples 48 kHz -> 16 kHz once to Drive`

`│   ├── run_train.py              # CLI entry point for fine-tuning`

`│   └── run_eval.py               # CLI entry point for evaluating Noisy, Spectral Gating, Pretrained, Fine-Tuned`

`├── results/`

`│   ├── metrics_summary.csv       # Tabular comparison of PESQ & STOI across all methods`

`│   └── figures/                  # Spectrogram comparisons (Noisy vs. Clean vs. Spectral Gating vs. Demucs)`

`├── requirements.txt              # Pinned dependencies (torch, torchaudio, denoiser, noisereduce, pesq, pystoi)`

`└── README.md                     # Documentation, setup guide, and reproducibility instructions`

### 2. Data Preprocessing Pipeline

1. **Single-Source Verification & Resampling:** Download the official dataset once to Google Drive (`/content/drive/MyDrive/SpeechEnhancement/data/`). If using Edinburgh Noisy Speech DB (48 kHz native), run a one-time preprocessing step using `torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)` and save the 16 kHz .wav files to Drive. Caching 16 kHz files once avoids resampling overhead on every training epoch and shrinks dataset size from ~4.9 GB to ~1.6 GB.
2. **Train / Validation / Test Split:** Test Set is strictly locked to the official test set (824 utterances from 2 unseen speakers p232 and p257 for Edinburgh) and never touched during training or hyperparameter selection. For Train / Validation, split the 28 training speakers by speaker ID (26 speakers for training, 2 speakers held out for validation) so validation measures generalization to unseen speakers.
3. **Segment Chunking & Normalization:** During training, slice or pad waveforms into fixed 3.0-second or 4.0-second segments (48,000 or 64,000 samples at 16 kHz) with a seeded random crop offset. Apply identical normalization scale factors (mean/std of the noisy signal) to both the noisy input and clean target so relative SNR is preserved.

### 3. Fine-Tuning & Training Strategy (Demucs U-Net DNN)

1. **Base Checkpoint:** Load `pretrained.dns48()` (or `dns64()`) from `facebookresearch/denoiser`.
2. **Fine-Tuning Protocol:** Stage 1 (Optional Warmup - 2 Epochs): Freeze the encoder (`model.encoder`) and LSTM bottleneck (`model.lstm`), fine-tuning only the U-Net decoder (`model.decoder`) at lr = 1e-4. Stage 2 (Full Fine-Tuning - 10 to 15 Epochs): Unfreeze all parameters and fine-tune end-to-end with a low learning rate (lr = 5e-5, AdamW optimizer, weight decay 1e-5, CosineAnnealingLR scheduler, and gradient clipping max_norm = 5.0) to prevent catastrophic forgetting.
3. **Loss Function (Perceptual + Time Domain):** Follow the official Demucs loss formulation: a combination of Time-Domain L1 Loss and Multi-Resolution STFT Loss (spectral convergence + log-magnitude L1 loss across FFT sizes [512, 1024, 2048]): `L_total = L_1(y, y_hat) + lambda * sum_m (L_sc^(m)(y, y_hat) + L_mag^(m)(y, y_hat))`. This directly optimizes both phase/waveform accuracy and time-frequency spectral sharpness, strongly correlating with PESQ and STOI improvements.

### 4. Evaluation Protocol (PESQ and STOI)

Every test utterance in the held-out test split will be evaluated across four conditions:

1. **Unprocessed Noisy Speech** (Reference lower bound)
2. **Spectral Gating Baseline** (Traditional DSP method)
3. **Pretrained Demucs (Zero-Shot)** (Out-of-the-box pretrained checkpoint before fine-tuning)
4. **Fine-Tuned Demucs** (Our adapted U-Net DNN model)
- **PESQ:** Computed at 16 kHz using the standard pesq package (`pesq(16000, clean_wav, enhanced_wav, 'wb')`), reporting Wideband MOS-LQO scores (range -0.5 to 4.5).
- **STOI:** Computed at 16 kHz using pystoi (`stoi(clean_wav, enhanced_wav, 16000, extended=False)`), reporting intelligibility scores (0.0 to 1.0).
- **Statistical Reporting:** Report Mean ± Standard Deviation across all test utterances, broken down by SNR level (2.5 dB, 7.5 dB, 12.5 dB, 17.5 dB), plus visual spectrogram comparisons and downloadable audio examples.

### 5. Google Colab, Google Drive & Antigravity 2.0 Workflow

- **Antigravity 2.0 (Local IDE):** Used to author, lint, and modularize all Python source files (src/, configs/, scripts/) and commit them to the workspace/Git repository.
- **Google Drive Storage Layout (/MyDrive/Speech-Enhancement-Data/):** Preprocessed 16 kHz dataset is archived as a single .tar or .zip file on Drive. At the start of a Colab session, the notebook copies the archive from Drive to Colab local NVMe SSD (/content/data_16k/) and extracts it in ~15 seconds (20x–50x faster than reading thousands of small .wav files over the Google Drive FUSE mount). Model checkpoints (best_model.pt, last_checkpoint.pt) and evaluation CSVs/plots are saved directly back to Google Drive after every epoch.

### 6. Reproducibility Requirements

- **Global Deterministic Seeding:** Set fixed seeds (seed = 42) across random, numpy, torch, and torch.cuda.manual_seed_all, plus deterministic DataLoader worker seeding (worker_init_fn).
- **Pinned Environment:** Lock exact package versions in requirements.txt (torch, torchaudio, denoiser==0.1.5, noisereduce==3.0.2, pesq==0.0.4, pystoi==0.4.1).
- **Config-Driven Execution:** Store all hyperparameters, SNR splits, and file paths in configs/config.yaml and save a snapshot of the config alongside every saved model checkpoint.

## Next Steps & Approval Checklist

1. **Dataset Selection:** Confirm whether to proceed with Candidate 1 (University of Edinburgh Noisy Speech Database / VoiceBank-DEMAND, DOI: 10.7488/ds/2117) or Candidate 2 (Microsoft MS-SNSD).
2. **Model & Architecture Plan:** Confirm using Spectral Gating as the traditional baseline and Fine-Tuned Demucs (dns48/dns64) as the main U-Net DNN model.
3. **Implementation Greenlight:** Proceed with creating the project structure and codebase upon confirmation.
