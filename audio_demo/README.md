# Offline Interactive Speech Enhancement Audio Demo

An interactive, zero-dependency, 100% offline web demonstration for comparing speech enhancement models on the official Edinburgh VoiceBank-DEMAND test benchmark.

---

## 1. Overview & Purpose

This demonstration provides an interactive audio listening environment for researchers, engineers, and evaluators to audition, inspect, and compare speech enhancement results without requiring any code execution, machine learning frameworks, or external network access.

The demo showcases speech enhancement performance across **5 audio conditions** on **4 representative test utterances** covering diverse acoustic environments, speaker genders, and signal-to-noise ratios (SNR from 2.5 dB to 17.5 dB). It also includes a side-by-side **A/B comparison controller** with seamless timestamp preservation and complete aggregate benchmark statistics ($N = 824$) from the formal evaluation.

---

## 2. Quick Start: 100% Offline Usage

This demo is designed to run entirely locally in any standard web browser using the `file://` protocol:

1. **Open the Demo:**
   - Double-click `index.html` in your file explorer, OR
   - Open your browser and navigate to:
     ```text
     file:///<path-to>/Speech-Enhancement/audio_demo/index.html
     ```
2. **Requirements:**
   - **No web server required** (no `http-server`, `python -m http.server`, or Node.js runtime needed).
   - **No internet connection required** (zero CDN dependencies, zero web fonts, zero remote tracking).
   - **No CORS issues**: Audio files are loaded via standard relative paths (`audio/<utterance_id>/<condition>.wav`) directly into native HTML5 `<audio>` elements, bypassing XMLHttp/fetch restrictions under `file://`.

---

## 3. The Five Audio Conditions

For every utterance, 5 synchronized audio conditions are provided (16 kHz, 16-bit PCM Mono WAV):

| Condition | Role | Technical Description |
| :--- | :--- | :--- |
| **1. Clean Speech Reference** (`clean.wav`) | Ground-Truth Target | Original unprocessed speech recorded in an anechoic studio chamber. Serves as the ground-truth target for objective perceptual quality (PESQ) and intelligibility (STOI). |
| **2. Noisy Input Mixture** (`noisy.wav`) | Unprocessed Baseline | Clean speech artificially degraded with real-world acoustic background noise recorded in the DEMAND database at specified SNR levels (2.5, 7.5, 12.5, 17.5 dB). |
| **3. Stationary Spectral Gating** (`spectral_gating.wav`) | Classical DSP Baseline | Traditional DSP baseline using non-neural stationary spectral gating (`scipy` / `noisereduce` STFT mask). Estimates stationary noise floor and attenuates stationary bands, exhibiting musical noise artifacts and high speech distortion under non-stationary noise. |
| **4. Demucs DNS48 Zero-Shot** (`dns48_zeroshot.wav`) | Pretrained Baseline | Out-of-the-box pretrained causal Demucs model (`dns48`, hidden=48, 4-layer waveform U-Net + bidirectional LSTM bottleneck) trained on the Microsoft DNS Challenge dataset without adaptation to VoiceBank-DEMAND. |
| **5. Demucs DNS48 Fine-Tuned** (`dns48_finetuned.wav`) | Project Model | Domain-adapted Demucs `dns48` fine-tuned for 15 epochs on VoiceBank-DEMAND 28-speaker training set using multi-resolution STFT loss + L1 waveform loss. Best checkpoint selected via validation PESQ at Epoch 14 (`best_val_pesq.th`). |

---

## 4. Representative Test Utterances

The demo includes 4 carefully selected test utterances representing the full distribution of noise environments, genders, and SNR levels in the official 824-utterance VoiceBank-DEMAND test set (unseen speakers `p232` and `p257`):

### Summary Table

| Utterance ID | Speaker | Noise Environment | SNR Level | Duration | Samples | File Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`p232_023`** | Male (`p232`) | `psquare` (Public Square) | 2.5 dB (Severe) | 9.77s | 156,302 | 312,648 bytes |
| **`p257_008`** | Female (`p257`) | `cafe` (Cafeteria) | 7.5 dB (Moderate) | 5.58s | 89,330 | 178,704 bytes |
| **`p232_012`** | Male (`p232`) | `living` (Living Room) | 12.5 dB (Mild) | 4.34s | 69,371 | 138,786 bytes |
| **`p257_014`** | Female (`p257`) | `office` (Office) | 17.5 dB (High SNR) | 3.69s | 59,080 | 118,204 bytes |

### Per-Utterance Objective Metrics

#### 1. Utterance `p232_023` (Male, `psquare` Public Square, 2.5 dB SNR)
- *Acoustic context:* Severe non-stationary outdoor crowd babble, echoing chatter, and footstep noise.
- *Evaluation metrics:*
  - **Clean Reference:** PESQ: Reference Target | STOI: 1.0000
  - **Noisy Input:** PESQ: 1.3108 | STOI: 0.9366 | $\Delta\text{PESQ}_\text{noisy}$: 0.0000 | $\Delta\text{PESQ}_\text{zs}$: -0.9518
  - **Spectral Gating:** PESQ: 1.1323 | STOI: 0.8165 | $\Delta\text{PESQ}_\text{noisy}$: -0.1785 | $\Delta\text{PESQ}_\text{zs}$: -1.1303
  - **DNS48 Zero-Shot:** PESQ: 2.2626 | STOI: 0.9671 | $\Delta\text{PESQ}_\text{noisy}$: +0.9518 | $\Delta\text{PESQ}_\text{zs}$: 0.0000
  - **DNS48 Fine-Tuned:** **PESQ: 2.5008** | **STOI: 0.9675** | **$\Delta\text{PESQ}_\text{noisy}$: +1.1900** | **$\Delta\text{PESQ}_\text{zs}$: +0.2382**

#### 2. Utterance `p257_008` (Female, `cafe` Cafeteria, 7.5 dB SNR)
- *Acoustic context:* Moderate indoor cafeteria background with multi-speaker babble, clattering dishes, and cutlery transients.
- *Evaluation metrics:*
  - **Clean Reference:** PESQ: Reference Target | STOI: 1.0000
  - **Noisy Input:** PESQ: 1.2496 | STOI: 0.9045 | $\Delta\text{PESQ}_\text{noisy}$: 0.0000 | $\Delta\text{PESQ}_\text{zs}$: -0.6555
  - **Spectral Gating:** PESQ: 1.0906 | STOI: 0.7877 | $\Delta\text{PESQ}_\text{noisy}$: -0.1590 | $\Delta\text{PESQ}_\text{zs}$: -0.8145
  - **DNS48 Zero-Shot:** PESQ: 1.9051 | STOI: 0.9325 | $\Delta\text{PESQ}_\text{noisy}$: +0.6555 | $\Delta\text{PESQ}_\text{zs}$: 0.0000
  - **DNS48 Fine-Tuned:** **PESQ: 2.2039** | **STOI: 0.9531** | **$\Delta\text{PESQ}_\text{noisy}$: +0.9543** | **$\Delta\text{PESQ}_\text{zs}$: +0.2988**

#### 3. Utterance `p232_012` (Male, `living` Living Room, 12.5 dB SNR)
- *Acoustic context:* Domestic reverberant room acoustics with steady HVAC low-frequency hum.
- *Evaluation metrics:*
  - **Clean Reference:** PESQ: Reference Target | STOI: 1.0000
  - **Noisy Input:** PESQ: 2.0530 | STOI: 0.9654 | $\Delta\text{PESQ}_\text{noisy}$: 0.0000 | $\Delta\text{PESQ}_\text{zs}$: -0.6551
  - **Spectral Gating:** PESQ: 1.1952 | STOI: 0.8279 | $\Delta\text{PESQ}_\text{noisy}$: -0.8578 | $\Delta\text{PESQ}_\text{zs}$: -1.5129
  - **DNS48 Zero-Shot:** PESQ: 2.7081 | STOI: 0.9648 | $\Delta\text{PESQ}_\text{noisy}$: +0.6551 | $\Delta\text{PESQ}_\text{zs}$: 0.0000
  - **DNS48 Fine-Tuned:** **PESQ: 3.1346** | **STOI: 0.9739** | **$\Delta\text{PESQ}_\text{noisy}$: +1.0816** | **$\Delta\text{PESQ}_\text{zs}$: +0.4265**

#### 4. Utterance `p257_014` (Female, `office` Office, 17.5 dB SNR)
- *Acoustic context:* High SNR office environment with subtle ventilation and keyboard typing transients.
- *Evaluation metrics:*
  - **Clean Reference:** PESQ: Reference Target | STOI: 1.0000
  - **Noisy Input:** PESQ: 3.3299 | STOI: 0.9944 | $\Delta\text{PESQ}_\text{noisy}$: 0.0000 | $\Delta\text{PESQ}_\text{zs}$: -0.0857
  - **Spectral Gating:** PESQ: 1.1785 | STOI: 0.8065 | $\Delta\text{PESQ}_\text{noisy}$: -2.1514 | $\Delta\text{PESQ}_\text{zs}$: -2.2371
  - **DNS48 Zero-Shot:** PESQ: 3.4156 | STOI: 0.9950 | $\Delta\text{PESQ}_\text{noisy}$: +0.0857 | $\Delta\text{PESQ}_\text{zs}$: 0.0000
  - **DNS48 Fine-Tuned:** **PESQ: 3.7012** | **STOI: 0.9961** | **$\Delta\text{PESQ}_\text{noisy}$: +0.3713** | **$\Delta\text{PESQ}_\text{zs}$: +0.2856**

---

## 5. Verified Official Aggregate Benchmark Results ($N = 824$)

The aggregate evaluation was conducted across all 824 test utterances of the official Edinburgh VoiceBank-DEMAND test set:

| Condition / Method | Role | Wideband PESQ (Mean ± Std) | PESQ Median | STOI (Mean ± Std) | STOI Median | ΔPESQ vs Noisy | ΔPESQ vs Zero-Shot | Success Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Noisy Input** | Unprocessed | 1.9719 ± 0.7532 | 1.7986 | 0.9211 ± 0.0709 | 0.9403 | — | -0.5484 (-21.8%) | 824 / 824 (100%) |
| **Spectral Gating** | DSP Baseline | 1.4232 ± 0.2848 | 1.3502 | 0.8234 ± 0.0655 | 0.8320 | -0.5487 (-27.8%) | -1.0971 (-43.5%) | 824 / 824 (100%) |
| **DNS48 Zero-Shot** | Pretrained | 2.5203 ± 0.6258 | 2.5263 | 0.9300 ± 0.0638 | 0.9476 | +0.5484 (+27.8%) | — | 824 / 824 (100%) |
| **DNS48 Fine-Tuned** | **Project Model** | **2.9180 ± 0.6483** | **3.0316** | **0.9480 ± 0.0507** | **0.9645** | **+0.9461 (+48.0%)** | **+0.3977 (+15.8%)** | **824 / 824 (100%)** |

### Statistical Hypothesis Testing Summary (Phase 4D Confirmatory Analysis)

> Aggregate test-set statistics over all $N = 824$ paired official test utterances (DNS48 Fine-Tuned vs. DNS48 Zero-Shot), sourced from `results/statistical_analysis/primary_and_secondary_tests.csv`. These values are not computed from the 4 demonstration samples.

- **Co-Primary Hypothesis 1 (PESQ Fine-Tuned > Zero-Shot):**
  - **Primary Exact Paired Sign Test:** **748 / 824 wins (90.78%)**, $p_{\text{raw}} = 1.21 \times 10^{-139}$, $p_{\text{Holm}} = 2.42 \times 10^{-139}$ (statistically significant at $\alpha = 0.01$)
  - **Complementary Wilcoxon Signed-Rank Test:** $Z = 22.12$, $p_{\text{raw}} = 1.87 \times 10^{-108}$, $p_{\text{Holm}} = 3.73 \times 10^{-108}$, Hodges-Lehmann median shift $= +0.4036$ ($95\%$ CI $[+0.3822, +0.4248]$)
  - **Secondary Paired Student's $t$-Test:** Mean paired gain $= +0.3977$ ($95\%$ CI $[+0.3735, +0.4214]$), $t(823) = 32.24$, $p_{\text{raw}} = 4.28 \times 10^{-148}$, $p_{\text{Holm}} = 8.55 \times 10^{-148}$
  - **Effect sizes:** Matched-pairs rank-biserial $r_{\text{rb}} = +0.8897$, Cohen's $d_z = 1.1231$ (Large effect)

- **Co-Primary Hypothesis 2 (STOI Fine-Tuned > Zero-Shot):**
  - **Primary Exact Paired Sign Test:** **669 / 824 wins (81.19%)**, $p_{\text{raw}} = 3.72 \times 10^{-81}$, $p_{\text{Holm}} = 3.72 \times 10^{-81}$ (statistically significant at $\alpha = 0.01$)
  - **Complementary Wilcoxon Signed-Rank Test:** $Z = 19.01$, $p_{\text{raw}} = 1.37 \times 10^{-80}$, $p_{\text{Holm}} = 1.37 \times 10^{-80}$, Hodges-Lehmann median shift $= +0.0136$ ($95\%$ CI $[+0.0117, +0.0155]$)
  - **Secondary Paired Student's $t$-Test:** Mean paired gain $= +0.0179$ ($95\%$ CI $[+0.0161, +0.0198]$), $t(823) = 18.54$, $p_{\text{raw}} = 2.24 \times 10^{-64}$, $p_{\text{Holm}} = 2.24 \times 10^{-64}$
  - **Effect sizes:** Matched-pairs rank-biserial $r_{\text{rb}} = +0.7692$, Cohen's $d_z = 0.6458$ (Medium-to-large effect)

---

## 6. Directory Structure & File Integrity

```text
audio_demo/
├── index.html                       # Standalone offline web demo (HTML5/CSS3/Vanilla JS)
├── README.md                        # Documentation and verification guide (this file)
├── audio_integrity_manifest.json    # SHA-256 byte-for-byte integrity manifest
└── audio/
    ├── p232_023/                    # Public Square (psquare, 2.5 dB SNR, 9.77s)
    │   ├── clean.wav                # 312,648 bytes
    │   ├── noisy.wav                # 312,648 bytes
    │   ├── spectral_gating.wav      # 312,648 bytes
    │   ├── dns48_zeroshot.wav       # 312,648 bytes
    │   └── dns48_finetuned.wav      # 312,648 bytes
    ├── p257_008/                    # Cafeteria (cafe, 7.5 dB SNR, 5.58s)
    │   ├── clean.wav                # 178,704 bytes
    │   ├── noisy.wav                # 178,704 bytes
    │   ├── spectral_gating.wav      # 178,704 bytes
    │   ├── dns48_zeroshot.wav       # 178,704 bytes
    │   └── dns48_finetuned.wav      # 178,704 bytes
    ├── p232_012/                    # Living Room (living, 12.5 dB SNR, 4.34s)
    │   ├── clean.wav                # 138,786 bytes
    │   ├── noisy.wav                # 138,786 bytes
    │   ├── spectral_gating.wav      # 138,786 bytes
    │   ├── dns48_zeroshot.wav       # 138,786 bytes
    │   └── dns48_finetuned.wav      # 138,786 bytes
    └── p257_014/                    # Office (office, 17.5 dB SNR, 3.69s)
        ├── clean.wav                # 118,204 bytes
        ├── noisy.wav                # 118,204 bytes
        ├── spectral_gating.wav      # 118,204 bytes
        ├── dns48_zeroshot.wav       # 118,204 bytes
        └── dns48_finetuned.wav      # 118,204 bytes
```

### Integrity Verification Guarantee
- **Total audio files:** 20 WAV files
- **Total audio size:** Exactly 3,741,710 bytes
- **Format:** 16,000 Hz, 16-bit Linear PCM, Mono WAV
- All 20 files are byte-identical copies of the source evaluation output files from `results/audio_examples/` and are cryptographically verified in `audio_integrity_manifest.json` using SHA-256 hashes.

---

## 7. Portable Standalone Archive (`Speech_Enhancement_Offline_Audio_Demo.zip`)

For portability and self-contained distribution, the entire `audio_demo/` folder is packaged into a root-level ZIP archive:
`Speech_Enhancement_Offline_Audio_Demo.zip`

### Extraction Instructions
1. Download or copy `Speech_Enhancement_Offline_Audio_Demo.zip` to any local machine.
2. Extract the archive:
   - **Windows:** Right-click &rarr; *Extract All...*
   - **Linux / macOS:** `unzip Speech_Enhancement_Offline_Audio_Demo.zip`
3. Enter the extracted folder and double-click `index.html`.
4. Press Play on any condition to start listening (no autoplay; no installation or dependencies required).

---

## 8. Technical Design & Browser Compatibility

- **Pure Vanilla Architecture:** 100% standard HTML5, CSS3, and ES6 JavaScript. Zero external frameworks (no React, Vue, jQuery, Bootstrap, or Tailwind).
- **Single-Audition Enforcement:** When any player starts playback, all other active audio players immediately pause to prevent cacophony.
- **A/B Instant Switching:** The dedicated A/B toolbar allows listeners to switch between any two conditions (e.g., Noisy vs. Fine-Tuned, or Zero-Shot vs. Fine-Tuned) while seamlessly preserving the exact playback timestamp.
- **Browser Compatibility:** Designed for modern browsers with standard HTML5 `<audio>` WAV support. Automated offline verification was performed in headless Microsoft Edge only (all 20 files loaded via `file://`); Chrome, Firefox and Safari have not been tested.
