# Speech-Enhancement

Academic speech enhancement system reducing background noise while preserving perceptual speech quality and intelligibility.

## Project Configuration

- **Dataset:** University of Edinburgh Noisy Speech Database (`DOI: 10.7488/ds/2117`, 28-speaker VoiceBank-DEMAND), resampled to **16 kHz mono**.
- **Main Pretrained Model:** Facebook Research Causal Demucs (`dns48` checkpoint, `hidden=48`, 16 kHz waveform U-Net + LSTM).
- **Traditional Baseline:** Stationary Spectral Gating (`noisereduce`).
- **Loss Function:** Time-domain $L_1$ loss + Multi-Resolution STFT loss (`stft_sc_factor = 0.1`, `stft_mag_factor = 0.1`).
- **Evaluation Metrics:** Wideband PESQ (`16 kHz`, `'wb'`) and STOI (`16 kHz`, `extended=False`) across all 824 official test utterances.

## Directory Structure

```text
Speech-Enhancement/
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── scripts/
│   └── 04_smoke_test.py
└── src/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   └── demucs_loader.py
    ├── training/
    │   ├── __init__.py
    │   └── stft_loss.py
    └── utils/
        ├── __init__.py
        ├── audio_io.py
        └── seed.py
```

## Running the Step 1 Smoke Test (Google Colab)

1. Install dependencies without legacy Hydra conflicts:
   ```bash
   pip install --no-deps git+https://github.com/facebookresearch/denoiser.git
   pip install -r requirements.txt
   ```
2. Run **Stage 1 (Synthetic 16 kHz Smoke Test — no dataset download required)**:
   ```bash
   python scripts/04_smoke_test.py
   ```
3. Run **Stage 2 (Real Clean/Noisy Audio Pair & PESQ/STOI Check)** after dataset license verification:
   ```bash
   python scripts/04_smoke_test.py --clean_wav path/to/clean.wav --noisy_wav path/to/noisy.wav
   ```
