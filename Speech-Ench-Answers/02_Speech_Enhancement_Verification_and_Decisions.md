> **Document ID:** `02` | **Phase:** Phase 1B — Verification & Architectural Decisions | **Role:** Technical Verification  
> **Source Word Document:** `Speech_Enhancement_Verification_and_Decisions.docx` | **Repository Path:** `Speech-Ench-Answers/02_Speech_Enhancement_Verification_and_Decisions.md`

---
# Technical Verification & Final Decision Report: Speech-Enhancement

**Scope:** Verification of Dataset Selection (Edinburgh vs. MS-SNSD), Pretrained Demucs Checkpoints (dns48 / dns64 / master64), Training Strategy, Spectral Gating Baseline, and PESQ/STOI Evaluation.

## 1. DATASET DECISION: Edinburgh Noisy Speech Database vs. Microsoft MS-SNSD

We do not claim that "NoisySpeech" is an official umbrella taxonomy. Rather, it is an ambiguous shorthand in the assignment prompt that points directly to one of two official repositories: one whose official title starts with *"Noisy speech database..."* (Edinburgh), and one whose synthesizer script literally outputs a directory named `NoisySpeech_training` (Microsoft MS-SNSD).

### Head-to-Head Verification Matrix

| Verification Criterion | Candidate A: University of Edinburgh Noisy Speech Database | Candidate B: Microsoft MS-SNSD |
| :--- | :--- | :--- |
| Exact Official Name | Noisy speech database for training speech enhancement algorithms and TTS models (commonly cited as VoiceBank-DEMAND or Valentini-Botinhao) | Microsoft Scalable Noisy Speech Dataset (MS-SNSD) |
| Official Publisher & Source | University of Edinburgh, Centre for Speech Technology Research (CSTR), Cassia Valentini-Botinhao et al. (2016). Hosted on Edinburgh DataShare: https://datashare.ed.ac.uk/handle/10283/2791 (DOI: 10.7488/ds/2117). | Microsoft Corporation, Chandan K. A. Reddy et al. (Interspeech 2019, arXiv:1904.03017). Hosted on GitHub: https://github.com/microsoft/MS-SNSD. |
| Paired Audio Structure | Pre-mixed, fixed, sample-aligned pairs. Train: 11,572 clean/noisy pairs (28 VCTK speakers; 10 noises: 8 DEMAND + 2 artificial at 0, 5, 10, 15 dB SNR). Test: 824 clean/noisy pairs (2 unseen speakers p232, p257; 5 unseen DEMAND noises at 2.5, 7.5, 12.5, 17.5 dB SNR). | Unmixed raw source directories (clean_train, noise_train, clean_test, noise_test). Requires executing python noisyspeech_synthesizer.py with noisyspeech_synthesizer.cfg to synthesize pairs into NoisySpeech_training/ and CleanSpeech_training/. |
| Native Sample Rate & Format | 48,000 Hz (48 kHz), 16-bit PCM mono .wav. Requires downsampling to 16,000 Hz for 16 kHz models and Wideband PESQ. | 16,000 Hz (16 kHz) natively, 16-bit PCM mono .wav (sampling_rate: 16000 in noisyspeech_synthesizer.cfg). |
| Storage Requirements | Download ZIPs (48 kHz): clean_trainset_28spk_wav.zip (2.37 GB) + noisy_trainset_28spk_wav.zip (2.70 GB) + clean_testset_wav.zip (147 MB) + noisy_testset_wav.zip (162 MB) = ~5.38 GB. Resampled to 16 kHz: ~1.75 GB uncompressed. | Git Repo Clone: ~1.5 GB of raw unmixed .wav files. Synthesized Output: Variable depending on total_hours in noisyspeech_synthesizer.cfg (default total_hours: 1 is ~230 MB; 10 hours is ~2.3 GB). |
| Official License | Creative Commons Attribution 4.0 International (CC BY 4.0) — explicitly verified on Edinburgh DataShare (10.7488/ds/2117). | Code: MIT License. Data: Provided by Microsoft "AS IS" under original third-party corpus terms for academic/research use. |
| Suitability & Reproducibility | Highest benchmark reproducibility: Fixed train and test splits mean zero variation from synthetic mixing parameters; directly supported by facebookresearch/denoiser (dset=valentini). | High flexibility, lower cross-study standardization: Because pairs are synthesized via random noise/SNR sampling, test scores depend on exact .cfg settings and RNG state during synthesis. |

### Single Dataset Recommendation

- **Selected Dataset: University of Edinburgh Noisy Speech Database (DOI: 10.7488/ds/2117).**
- **Justification:** It provides pre-mixed, sample-aligned clean and noisy pairs with a universally standardized 824-utterance speaker-independent test split, is licensed under CC BY 4.0, fits easily within Google Drive and Colab SSD limits (~1.75 GB once resampled to 16 kHz), and has native configuration support inside `facebookresearch/denoiser`.

## 2. DEMUCS VERIFICATION (facebookresearch/denoiser)

All claims below are verified directly against the official Meta Research repository (https://github.com/facebookresearch/denoiser) and the accompanying paper (Défossez et al., "Real Time Speech Enhancement in the Waveform Domain", Interspeech 2020, arXiv:2006.12847).

### 1. Pretrained Checkpoint Details & Pretraining Dataset Differences

In `denoiser/pretrained.py`, the repository provides three causal Demucs checkpoints (`depth=5, kernel_size=8, stride=4, resample=4, causal=True`, 2-layer unidirectional LSTM bottleneck):

| Checkpoint Flag / Function | Hidden Channels (H) | Approx. Parameters | Official Pretraining Dataset | Suitability for Fine-Tuning on Edinburgh (valentini) |
| :--- | :--- | :--- | :--- | :--- |
| dns48 (pretrained.dns48()) | hidden=48 | ~18.9M | Trained exclusively on the Microsoft INTERSPEECH 2020 DNS Challenge dataset. | Optimal: Zero overlap with the Edinburgh (Valentini) dataset. Fast training and low VRAM footprint on Colab T4. |
| dns64 (pretrained.dns64()) | hidden=64 | ~33.5M | Trained exclusively on the Microsoft INTERSPEECH 2020 DNS Challenge dataset. | Excellent: Zero overlap with Edinburgh (Valentini); higher capacity (H=64) than dns48. |
| master64 (pretrained.master64()) | hidden=64 | ~33.5M | Trained on DNS Challenge + Edinburgh (Valentini) combined. | Avoid for fine-tuning: Because master64 was already trained on the Valentini dataset, using it as a starting point compromises clean domain-adaptation benchmarking. |

- **Checkpoint Selection:** We select `dns48` (or `dns64`) and explicitly reject `master64` so our zero-shot pretrained baseline (`dns48`) has never seen the Edinburgh VoiceBank-DEMAND corpus prior to our fine-tuning stage.

### 2. Sample Rate and Input Requirements

- **Sample Rate:** Hardcoded in `denoiser/demucs.py` and `conf/config.yaml` as `sample_rate: 16000` (16 kHz), single-channel (`chin: 1, chout: 1`).
- **Tensor Shape:** Expects 3D float32 waveform tensors of shape `[batch_size, 1, num_samples]`.
- **Internal Resampling & Normalization:** Inside `Demucs.forward()`, when `normalize=True` (default in pretrained models), the model normalizes the input waveform by its standard deviation (`std = mono.std(dim=-1, keepdim=True)`) before encoding and rescales the output by `std`. It also upsamples internally by `resample=4` via sinc interpolation before the encoder and downsamples by 4 at the output.
- **Compatibility with Selected Dataset:** Edinburgh Noisy Speech DB is 48 kHz mono .wav. Downsampling it to 16 kHz makes it 100% compatible (`denoiser/dsp.py` provides `convert_audio(wav, from_samplerate, to_samplerate=16000, channels=1)`, or we preprocess once via `torchaudio.transforms.Resample(48000, 16000)`).

### 3. Official Fine-Tuning Commands & Configuration

- **JSON Manifest Format (egs/):** The official dataset loader (`denoiser/data.py -> NoisyCleanSet`) reads two JSON files inside each split folder (`tr/, cv/, tt/`): `noisy.json` and `clean.json`, where each JSON is a list of `[wav_path, num_samples]` pairs generated via `python -m denoiser.audio path/to/wavs > noisy.json`.
- **Official Hydra Fine-Tuning Flags (conf/config.yaml):** `continue_pretrained=dns48` (or `dns64`), `demucs.hidden=48` (or 64), `dset=valentini`, and `stft_loss=True`.

`python train.py dset=valentini continue_pretrained=dns48 demucs.hidden=48 stft_loss=True`

## 3. TRAINING STRATEGY VERIFICATION (Official Defaults vs. Proposed Experimental Settings)

Below is a strict separation between what `facebookresearch/denoiser` sets as defaults in `conf/config.yaml` (designed for multi-GPU V100 clusters training from scratch for 500 epochs) and our proposed experimental overrides tailored for fine-tuning on a single Google Colab T4 GPU.

| Parameter / Component | Official facebookresearch/denoiser Default (conf/config.yaml) | Proposed Experimental Setting (Colab Fine-Tuning) | Technical Justification |
| :--- | :--- | :--- | :--- |
| Train / Val / Test Split | make_valentini.sh uses all 28 train speakers in tr/ and points both cv/ (validation) and tt/ (test) to the 2-speaker test set (p232, p257). | Train (tr/): 26 speakers (10,729 utterances). Validation (cv/): 2 held-out train speakers (p226, p287 — 843 utterances). Test (tt/): Official 2 test speakers (p232, p257 — 824 utterances). | Fixes validation leakage: The official script using the test set as cv/ leaks test performance into model selection (best.th). Holding out 2 training speakers (p226, p287) guarantees strict speaker independence across Train, Val, and Test. |
| Segment Length & Stride | segment: 4.0 seconds (64,000 samples at 16 kHz), stride: 1.0 second (pad: True). | segment: 4.0 seconds, stride: 2.0 seconds (pad: True). | Increasing stride from 1.0s to 2.0s reduces overlapping crops per utterance by 50%, cutting epoch duration in half on a single Colab GPU while still covering all audio. |
| Loss Function | loss: l1 (Time-domain L1 loss). stft_loss: False by default in config.yaml, but enabled (stft_loss: True) in paper experiments with stft_sc_factor: 0.5, stft_mag_factor: 0.5 (FFT sizes [512, 1024, 2048], hop sizes [50, 120, 240], win lengths [240, 600, 1200]). | loss: l1 + stft_loss: True (stft_sc_factor: 0.5, stft_mag_factor: 0.5), matching the official denoiser/stft_loss.py (MultiResolutionSTFTLoss). | In arXiv:2006.12847, combining L1 waveform loss with Multi-Resolution STFT loss significantly improves PESQ by penalizing high-frequency spectral artifacts. |
| Optimizer & Learning Rate | optim: adam, lr: 3e-4, beta2: 0.999 (for training from scratch). | optim: adam, lr: 5e-5 (or 1e-4), beta2: 0.999 (end-to-end fine-tuning via continue_pretrained). | A 3x–6x lower learning rate (5e-5 to 1e-4) prevents large initial gradients from destroying the pretrained dns48 weights. |
| Batch Size | batch_size: 64 (across 8 GPUs). | batch_size: 8 or 16 (on 1x Colab T4 15 GB GPU). | Fits comfortably inside 15 GB VRAM (~4.5 GB VRAM used at batch_size=16 for dns48). |
| Epoch Count | epochs: 500 (training from scratch). | epochs: 15 to 20 (fine-tuning from dns48). | Since dns48 is already converged on 16 kHz speech enhancement, domain adaptation on 26 speakers converges within 10–20 epochs (~45–60 minutes total on a Colab T4). |
| Checkpointing Strategy | denoiser/solver.py automatically saves checkpoint.th (latest epoch state, model, optimizer, history) and best.th (lowest validation loss) after every epoch. Automatically resumes unless restart=1 is passed. | Configure output/checkpoint directory on Google Drive (/content/drive/MyDrive/Speech-Enhancement/checkpoints/) and follow solver.py checkpoint.th (resume) + best.th (best validation loss/PESQ) pattern. | Ensures automatic recovery if Google Colab disconnects mid-training. |

## 4. SPECTRAL GATING BASELINE SPECIFICATION

To ensure our traditional DSP baseline is 100% reproducible, non-arbitrary, and evaluated on the exact same footing as Demucs:

1. **Implementation Library & Reference:** `noisereduce` (`pip install noisereduce==3.0.2`, Sainburg et al., https://github.com/timsainb/noisereduce).
2. **Exact Algorithm & Parameter Specification (16 kHz Audio):** Stationary Spectral Gating (`stationary=True`), which estimates the noise profile across the utterance without requiring an external isolated noise reference clip: `sr=16000, n_fft=512 (32 ms window), win_length=512, hop_length=128 (8 ms hop, 75% overlap, Hann window), n_std_thresh_stationary=1.5, prop_decrease=1.0, time_constant_s=2.0, freq_mask_smooth_hz=500, time_mask_smooth_ms=50`.
3. **Evaluation Procedure:** Run `nr.reduce_noise(...)` on all 824 test utterances (p232, p257), verify output length matches `len(clean_wav_16k)` exactly, and compute Wideband PESQ and STOI on every utterance.

## 5. EVALUATION VERIFICATION (PESQ & STOI)

Verified against ITU-T Recommendation P.862.2, `pesq` (https://github.com/ludlows/python-pesq), `pystoi` (https://github.com/mpariente/pystoi), and `facebookresearch/denoiser/denoiser/evaluate.py`:

- **PESQ (pesq==0.0.4):** Only supports 8,000 Hz (`mode='nb'`) or 16,000 Hz (`mode='wb'`). Passing 48,000 Hz raises a `ValueError: fs must be 8000 or 16000`. Since our entire pipeline and Demucs operate at 16,000 Hz, we evaluate using Wideband PESQ (`pesq(16000, clean_np, enhanced_np, 'wb')`), which outputs raw MOS-LQO scores in [-0.5, 4.5].
- **STOI (pystoi==0.4.1):** Supports arbitrary sample rates by internally resampling to 10,000 Hz (`stoi(clean_np, enhanced_np, 16000, extended=False)`), returning a scalar intelligibility score in [0.0, 1.0].
- **Clean / Noisy / Enhanced Alignment Verification:** In the Edinburgh Noisy Speech Database (10.7488/ds/2117), each clean/noisy file pair has the exact same sample count at 48 kHz and remains sample-aligned after identical polyphase resampling to 16 kHz. In Demucs.forward(noisy), the model pads the input waveform to match the strided convolution receptive field and slices the output back to x[..., :length], guaranteeing enhanced.shape == noisy.shape == clean.shape. Our evaluation script will explicitly assert exact length equality and check for NaNs before calling pesq and stoi.

## 6. FINAL DECISION SUMMARY

| Decision Area | Locked Specification |
| :--- | :--- |
| 1. Selected Dataset | University of Edinburgh Noisy Speech Database (VoiceBank-DEMAND, 28-speaker version) — Official source: Edinburgh DataShare (DOI: 10.7488/ds/2117), CC BY 4.0 license, downsampled once from 48 kHz to 16 kHz (~1.75 GB). |
| 2. Speaker-Independent Splits | Train: 26 speakers (10,729 pairs) \| Validation: 2 held-out train speakers (p226, p287 — 843 pairs) \| Test: Official 2 unseen test speakers (p232, p257 — 824 pairs). |
| 3. Main Pretrained Model | Causal Demucs (dns48 checkpoint) from facebookresearch/denoiser (hidden=48, 16 kHz waveform U-Net + LSTM, pretrained strictly on Microsoft DNS so there is zero data leakage on Edinburgh). |
| 4. Traditional Baseline | Stationary Spectral Gating (noisereduce==3.0.2, sr=16000, n_fft=512, hop_length=128, n_std_thresh_stationary=1.5). |
| 5. Final Training Strategy | End-to-end fine-tuning from dns48 (continue_pretrained=dns48) for 15–20 epochs on 4.0s segments (stride=2.0s), batch_size=16, Adam optimizer (lr=5e-5, beta2=0.999), optimizing L1 Waveform Loss + Multi-Resolution STFT Loss (stft_sc_factor=0.5, stft_mag_factor=0.5), saving checkpoint.th and best.th to Google Drive. |
| 6. Evaluation Benchmark | Compare 4 conditions on all 824 test utterances at 16 kHz using Wideband PESQ (mode="wb") and STOI (extended=False): (1) Unprocessed Noisy Speech, (2) Spectral Gating Baseline, (3) Pretrained Demucs dns48 Zero-Shot, (4) Fine-Tuned Demucs dns48. |

### Remaining Uncertainties

1. **Course Instructor Dataset Preference:** While Edinburgh (10.7488/ds/2117) is the strongest technical choice, if the course instructor specifically distributed or mandated the Microsoft MS-SNSD repository, MS-SNSD can be substituted as our single dataset source.
