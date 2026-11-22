> **Document ID:** `03` | **Phase:** Phase 1C — Revised Implementation Plan | **Role:** System Architecture & Planning  
> **Source Word Document:** `Speech_Enhancement_Revised_Implementation_Plan.docx` | **Repository Path:** `Speech-Ench-Answers/03_Speech_Enhancement_Revised_Implementation_Plan.md`

---
# Revised Practical Implementation Plan & Technical Risk Analysis: Speech-Enhancement

**Approved Configuration:** Dataset: University of Edinburgh Noisy Speech Database (DOI: 10.7488/ds/2117, 48 kHz preserved + 16 kHz mono version)  |  Main Model: Facebook Research Demucs dns48  |  Baseline: Stationary Spectral Gating (noisereduce)  |  Evaluation: Wideband PESQ & STOI at 16 kHz across all 824 official test utterances

## 0. Integration of Mandatory Corrections

Before detailing the implementation modules, here is how all five mandatory corrections are built directly into our engineering plan:

1. **Empirical Verification of Speaker IDs & Utterance Counts:** We do not hardcode assumed utterance counts for training or validation splits. Instead, our split script (`scripts/03_create_splits.py`) scans the actual extracted filenames (`<speaker_id>_<utterance_id>.wav`), parses the speaker prefix (`pXXX`), verifies 1-to-1 filename and sample-length parity between clean and noisy directories, exports an exact empirical speaker audit (`results/speaker_audit.json`), and programmatically enforces strict speaker independence: Speakers_train ∩ Speakers_val = ∅, Speakers_train ∩ Speakers_test = ∅, and Speakers_val ∩ Speakers_test = ∅.
2. **Corrected Official STFT Loss Weighting (0.1, not 0.5):** While `conf/config.yaml` in `facebookresearch/denoiser` sets generic placeholders (`stft_loss: False, stft_sc_factor: .5, stft_mag_factor: .5`), the official causal Valentini launcher (`launch_valentini.sh`) explicitly overrides them to `stft_loss=True stft_sc_factor=0.1 stft_mag_factor=0.1`. Our configuration locks in `stft_sc_factor: 0.1` and `stft_mag_factor: 0.1` as the strict default.
3. **Experimental Assumptions vs. Guaranteed Facts:** Hyperparameters (`epochs: 15–20, batch_size: 8–16, lr: 5e-5`), training duration, and GPU VRAM usage are treated strictly as initial experimental assumptions. Our pipeline includes an explicit 1-epoch profiling/benchmark step on Google Colab to measure actual step latency and peak GPU memory (`torch.cuda.max_memory_allocated()`) before running a multi-epoch schedule.
4. **Compatibility of Archived facebookresearch/denoiser with Current Google Colab:** facebookresearch/denoiser (archived in 2023) pins `hydra-core<1.0 (0.11.3)` and `omegaconf<1.5` for `train.py`, which fails under modern Python 3.10+/3.11+ on Google Colab. Additionally, `denoiser/stft_loss.py` calls `torch.stft()` without `return_complex=True`, which raises a RuntimeError in PyTorch 2.x. Solution: We install denoiser with `--no-deps` (plus `julius`) to use the official Demucs model class and `pretrained.dns48()` checkpoint loader directly in PyTorch without invoking legacy Hydra, and we provide a PyTorch 2.x–compatible MultiResolutionSTFTLoss (`return_complex=True`) verified in `04_test_compatibility.py`.
5. **Item-Level License Verification (DOI: 10.7488/ds/2117):** Rather than inferring license terms from DataShare metadata defaults, `scripts/01_download_dataset.py` downloads the actual item-level license bitstream directly from handle 10283/2791 (`https://datashare.ed.ac.uk/bitstream/handle/10283/2791/license_text`) alongside `README.txt`, stores them with the raw archives, and prints the verbatim license text into the verification log before audio processing begins.

## 1. Project Folder Structure

All code will be authored locally in your Antigravity workspace (`Speech-Enhancement`) and executed on Google Colab with Google Drive mounted for persistent dataset, checkpoint, and result storage:

`Speech-Enhancement/`

`├── configs/`

`│   └── config.yaml                     # Central config (sr=16000, stft_sc_factor=0.1, stft_mag_factor=0.1, paths)`

`├── src/`

`│   ├── __init__.py`

`│   ├── utils/`

`│   │   ├── seed.py                     # Deterministic seeding (random, numpy, torch, cuda, DataLoader workers)`

`│   │   └── audio_io.py                 # Safe WAV load/save, 48k->16k resampling, sample-alignment assertions`

`│   ├── data/`

`│   │   ├── preprocess.py               # Extracts 48k ZIPs, verifies paired files, resamples to 16 kHz mono`

`│   │   ├── splits.py                   # Empirical speaker/utterance count audit & speaker-independent split builder`

`│   │   └── dataset.py                  # PyTorch Dataset (segment chunking, stride, padding)`

`│   ├── models/`

`│   │   └── demucs_loader.py            # Loads official pretrained dns48 checkpoint & verifies architecture`

`│   ├── baselines/`

`│   │   └── spectral_gating.py          # Stationary Spectral Gating baseline using noisereduce`

`│   ├── training/`

`│   │   ├── stft_loss.py                # PyTorch 2.x Multi-Resolution STFT Loss (sc_factor=0.1, mag_factor=0.1)`

`│   │   └── trainer.py                  # Fine-tuning engine, validation loop, Drive checkpointing & auto-resume`

`│   └── evaluation/`

`│       └── evaluator.py                # 16 kHz Wideband PESQ & STOI evaluation across all 824 test utterances`

`├── scripts/`

`│   ├── 01_download_dataset.py          # Fetches license_text, README.txt, and official 48k ZIPs to Drive`

`│   ├── 02_resample_16k.py              # Preserves original 48k ZIPs & creates 16 kHz mono WAV dataset`

`│   ├── 03_create_splits.py             # Audits speaker IDs/counts & writes speaker-independent tr/cv/tt manifests`

`│   ├── 04_test_compatibility.py        # Verifies Colab PyTorch 2.x, dns48 weights, STFT loss, pesq & pystoi`

`│   ├── 05_evaluate_baselines.py        # Evaluates Noisy, Spectral Gating, and Pretrained dns48 on 824 test files`

`│   ├── 06_finetune_demucs.py           # Runs fine-tuning of dns48 with epoch checkpointing to Google Drive`

`│   └── 07_evaluate_and_report.py       # Evaluates Fine-Tuned dns48 and compiles 4-way comparison tables/plots`

`├── notebooks/`

`│   └── Speech_Enhancement_Colab.ipynb  # End-to-end Google Colab notebook (Drive mount, environment setup, execution)`

`├── requirements.txt                    # Colab-tested dependencies`

`└── README.md                           # Reproducibility and execution instructions`

## 2. Step-by-Step Technical Implementation Plan

### Step 1: Dataset Download, Item-Level License Verification & Extraction (scripts/01_download_dataset.py)

- **Official Bitstreams (DOI: 10.7488/ds/2117 / Handle 10283/2791):** license_text, README.txt, trainset_28spk_txt.zip, testset_txt.zip, clean_trainset_28spk_wav.zip (~2.37 GB), noisy_trainset_28spk_wav.zip (~2.70 GB), clean_testset_wav.zip (~147 MB), and noisy_testset_wav.zip (~162 MB).
- **Execution & Preservation:** Downloads license_text and README.txt first, saves them to `/content/drive/MyDrive/Speech-Enhancement/data/raw_48k_zips/`, and prints license_text verbatim. Downloads the 4 audio .zip archives and 2 log .zip archives using resumable HTTP (`curl -L -C - --retry 5`) into Google Drive, preserving the original 48 kHz dataset permanently. Extracts archives to Colab local NVMe SSD (`/content/tmp_48k/`) for fast local I/O during resampling.

### Step 2: Audio Preprocessing & 16 kHz Mono Resampling (scripts/02_resample_16k.py)

- **Verification Before Resampling:** Verifies that every .wav file in clean_trainset_28spk_wav has a matching filename in noisy_trainset_28spk_wav (and likewise for test sets). Asserts `sr == 48000, channels == 1`, and `clean_48k.shape == noisy_48k.shape` for every pair.
- **Resampling & Format:** Uses `torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)` to downsample every file to 16,000 Hz mono and saves as 16-bit PCM .wav. Asserts `clean_16k.shape == noisy_16k.shape` after resampling.
- **Google Drive Packaging:** Packs the complete 16 kHz dataset into `/content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar` (~1.75 GB). Subsequent Colab sessions unpack this tar archive directly into Colab local SSD (`/content/data_16k/`) in ~10–15 seconds, avoiding Google Drive FUSE bottlenecks.

### Step 3: Speaker Audit & Speaker-Independent Split Generation (scripts/03_create_splits.py)

- **Empirical Audit:** Parses all filenames (`pXXX_YYY.wav`) in the training and test directories and cross-references them against trainset_28spk_txt and testset_txt. Outputs `results/speaker_audit.json` listing every discovered speaker ID, exact utterance count, total duration in seconds, and SNR distribution.
- **Speaker-Independent Partitioning:** Test Split (tt) is locked to the official test set speakers (p232 and p257, verified 824 utterances). Validation Split (cv) holds out 2 complete speakers (p226 and p287) from the 28 training speakers after reporting their verified utterance counts. Train Split (tr) uses the remaining 26 training speakers. Programmatically asserts zero speaker overlap across tr, cv, and tt, and writes JSON manifests (`manifests/{tr,cv,tt}/{noisy,clean}.json`).

### Step 4: Demucs Installation & Colab Compatibility Test (scripts/04_test_compatibility.py)

`pip install --no-deps git+https://github.com/facebookresearch/denoiser.git`

`pip install julius noisereduce==3.0.2 pesq==0.0.4 pystoi==0.4.1 soundfile pyyaml tqdm matplotlib pandas`

- **Verification Checks:** (1) Verifies Python version, PyTorch version, and CUDA GPU availability. (2) Loads `model = pretrained.dns48()` and verifies `sample_rate == 16000, chin == 1, chout == 1, hidden == 48`. (3) Runs a synthetic 16 kHz batch [2, 1, 64000] through eval and train modes, executes L1 + MultiResolutionSTFTLoss(sc_factor=0.1, mag_factor=0.1) backward pass, verifies non-NaN gradients, and reports peak GPU memory. (4) Runs a smoke test of `pesq(16000, ref, deg, 'wb')` and `stoi(ref, deg, 16000, extended=False)`.

### Step 5: Spectral Gating Baseline & Pretrained dns48 Inference Test (scripts/05_evaluate_baselines.py)

Before any fine-tuning, evaluates the first three conditions across all 824 official test utterances at 16 kHz and saves results to `/content/drive/MyDrive/Speech-Enhancement/results/baseline_and_zeroshot_metrics.csv`:

1. **Unprocessed Noisy Speech:** Direct comparison of noisy_testset_16k vs. clean_testset_16k.
2. **Stationary Spectral Gating Baseline (src/baselines/spectral_gating.py):** Uses `noisereduce.reduce_noise(y=noisy_np, sr=16000, stationary=True, n_fft=512, win_length=512, hop_length=128, n_std_thresh_stationary=1.5, prop_decrease=1.0)`. Asserts output sample length equals input sample length.
3. **Pretrained Demucs dns48 (Zero-Shot Inference):** Runs `pretrained.dns48().cuda().eval()` with `torch.no_grad()` on each full-length test utterance ([1, 1, T]), asserting exact output length parity.

### Step 6: Fine-Tuning Pipeline in Google Colab (scripts/06_finetune_demucs.py)

- **Model Initialization:** Starts from official `pretrained.dns48()`.
- **Loss Function (Corrected Official Valentini Configuration):** Time-domain L1 loss + Multi-Resolution STFT loss (`fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]`) with `stft_sc_factor = 0.1` and `stft_mag_factor = 0.1`: L_total = ||y - y_hat||_1 + 0.1 * L_sc(y, y_hat) + 0.1 * L_mag(y, y_hat).
- **Audio Framing & Normalization:** Slices training utterances into segment = 4.0s (64,000 samples) windows with stride = 2.0s (pad = True). Uses Demucs native per-utterance standard deviation normalization (normalize=True).
- **Experimental Training Settings & 1-Epoch Profiling:** Initial experimental settings: `optim = Adam(lr=5e-5, betas=(0.9, 0.999)), batch_size = 16, epochs = 15, max_norm = 5.0`. Evaluates validation loss and validation Wideband PESQ on the speaker-independent validation split (cv) at Epoch 0 (before weight updates) and after every epoch. Logs actual epoch duration and peak GPU VRAM after Epoch 1.

### Step 7: PESQ / STOI Evaluation & Google Drive Storage (scripts/07_evaluate_and_report.py)

- **Checkpointing on Google Drive (/content/drive/MyDrive/Speech-Enhancement/checkpoints/):** Saves `checkpoint.th` after every epoch for automatic resume if Colab disconnects, plus `best_val_loss.th` and `best_val_pesq.th`.
- **Final 4-Way Evaluation on All 824 Test Utterances:** Evaluates `best_val_pesq.th` on all 824 test utterances using `pesq(16000, clean, enhanced, 'wb')` and `stoi(clean, enhanced, 16000, extended=False)`. Produces `final_4way_comparison.csv` (Noisy, Spectral Gating, Pretrained dns48, Fine-Tuned dns48), `snr_breakdown_comparison.csv` (2.5, 7.5, 12.5, 17.5 dB SNR), training curves, spectrogram plots, and enhanced audio samples on Google Drive.

## 3. Remaining Technical Risks & Mitigations

| Risk | Cause | Mitigation Strategy |
| :--- | :--- | :--- |
| 1. PyTorch 2.x torch.stft Complex Tensor Requirement | denoiser/stft_loss.py was written for PyTorch 1.x where torch.stft returned a [..., 2] real/imag tensor by default; PyTorch 2.x throws an error unless return_complex=True is passed. | Implement src/training/stft_loss.py using torch.stft(..., return_complex=True) and torch.abs(x_stft), with unit verification in 04_test_compatibility.py. |
| 2. torchaudio Backend Changes in PyTorch 2.x | Newer torchaudio releases on Colab default to soundfile or ffmpeg backends rather than legacy sox_io. | Use soundfile (sf.read / sf.write with explicit PCM_16 subtype) alongside torchaudio.transforms.Resample in src/utils/audio_io.py so audio I/O is backend-agnostic across Windows and Linux Colab. |
| 3. Fine-Tuning Degradation Relative to Zero-Shot dns48 | Because dns48 is already strongly pretrained, an overly aggressive learning rate can degrade validation PESQ below the Epoch 0 zero-shot score. | Evaluate Epoch 0 on the validation set before the first gradient step, use a conservative lr=5e-5 with gradient clipping (5.0), and initialize best_val_pesq with the Epoch 0 score so best_val_pesq.th is guaranteed to be at least as strong as zero-shot dns48. |
| 4. Colab Session Timeouts During Resampling or Training | Free Colab sessions can disconnect after idle periods or long runs. | Store raw ZIPs and the resampled edinburgh_16k.tar archive on Google Drive so preprocessing only ever runs once, and save checkpoint.th to Google Drive after every single epoch (~2–3 minutes per epoch). |
