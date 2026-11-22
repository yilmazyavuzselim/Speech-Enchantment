> **Document ID:** `04` | **Phase:** Phase 1D — Pre-Implementation Final Technical Review | **Role:** Pre-Implementation QA Review  
> **Source Word Document:** `Speech_Enhancement_Final_Technical_Review.docx` | **Repository Path:** `Speech-Ench-Answers/04_Speech_Enhancement_Final_Technical_Review.md`

---
# Final Pre-Implementation Technical Review: Speech-Enhancement

**Review Scope:** (1) Demucs Compatibility (Inference vs. Training), (2) Training Pipeline Reuse & Adaptation Matrix, (3) 7-Point Minimal Smoke Test Specification, (4) Data Integrity & Test Isolation, (5) Required Corrections, Remaining Risks, and Exact First Implementation Step.

## 1. Verified Technical Decisions

### A. Demucs Compatibility (Loading/Inference vs. Original Training Pipeline)

- **Model Loading & Inference (Compatible):** `denoiser/pretrained.py` downloads `https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns48-11decc9d8e3f0998.th` via `torch.hub.load_state_dict_from_url` and instantiates `Demucs(..., hidden=48, sample_rate=16000, chin=1, chout=1, causal=True, resample=4)`. Both `denoiser/demucs.py` and `denoiser/resample.py` rely strictly on standard `torch.nn` layers (Conv1d, ConvTranspose1d, GLU, ReLU, LSTM) and pure-tensor sinc interpolation. Installing denoiser with `pip install --no-deps` (plus `julius`) allows `pretrained.dns48()` to load and run forward/backward passes in modern Google Colab (Python 3.10+/3.11+, PyTorch 2.x).
- **Original Training Pipeline (Incompatible Out-of-the-Box):** Loading dns48 does NOT make facebookresearch/denoiser original training scripts (`train.py, denoiser/solver.py, denoiser/stft_loss.py, denoiser/audio.py`) compatible with current Google Colab. Those scripts fail due to three distinct breaking changes in PyTorch 2.x / Python 3.10+:
  - **1. Legacy Hydra Conflict:** `train.py` requires legacy `hydra-core==0.11.3` and `omegaconf<1.5`, which fail under Python 3.10+ (collections.MutableMapping removal and syntax incompatibilities).
  - **2. PyTorch 2.x torch.stft Breakage:** `denoiser/stft_loss.py` calls `torch.stft(..., return_complex=False)` (implicitly expecting a legacy [..., 2] real/imaginary tensor) and computes `torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-7)`. In PyTorch 2.x, non-complex torch.stft was removed, raising a RuntimeError.
  - **3. Removed torchaudio.info().si Struct:** `denoiser/audio.py (Audioset)` calls `torchaudio.info(file).si.length`, which relies on the removed PyTorch 1.x sox_io C++ SignalInfo struct (`AttributeError: 'AudioMetaData' object has no attribute 'si'`).

### B. Training Pipeline Component-by-Component Reuse & Adaptation Matrix

| Component | Reuse Official denoiser Code? | Required Adaptation in Our Custom Training Pipeline |
| :--- | :--- | :--- |
| 1. Model (Demucs) | Reuse 100% (denoiser/demucs.py & pretrained.py) | Instantiate via pretrained.dns48(pretrained=True). No architectural modifications so pretrained weights match 1-to-1. |
| 2. Loss (L1 + MultiResolutionSTFTLoss) | Adapt (src/training/stft_loss.py) | Replace legacy denoiser/stft_loss.py with a PyTorch 2.x implementation using torch.stft(..., return_complex=True) and torch.sqrt(x.real**2 + x.imag**2 + 1e-7) with identical parameters (fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], sc_factor=0.1, mag_factor=0.1) plus F.l1_loss(clean, estimate). |
| 3. Optimizer | Adapt (src/training/trainer.py) | Standard torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999)) with gradient clipping (torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)). |
| 4. Data Loader (NoisyCleanSet) | Adapt (src/data/dataset.py) | Replace denoiser/audio.py (torchaudio.info().si) with soundfile.info(path).frames and soundfile.read(path, dtype="float32"), preserving the exact sliding-window chunking logic (length = int(segment * 16000), stride = int(stride * 16000), pad = True). |
| 5. Checkpointing | Adapt (src/training/trainer.py) | Save standard PyTorch dictionaries ({epoch, model_state_dict, optimizer_state_dict, history, config}) via atomic write (tmp file + os.replace) to Google Drive (checkpoint.th, best_val_loss.th, best_val_pesq.th). |
| 6. Validation Loop | Adapt (src/training/trainer.py) | Evaluate full-length validation utterances (cv split, 2 unseen speakers) in model.eval() + torch.no_grad(), computing validation loss, Wideband PESQ (16000, "wb"), and STOI at Epoch 0 (before training) and after every epoch. |

### C. Minimal Smoke Test Specification (scripts/04_smoke_test.py)

Before any dataset-wide run or fine-tuning, the minimal smoke test must pass all 7 assertions on 1 real 16 kHz clean/noisy audio pair:

1. **Model Loading:** Load `model = pretrained.dns48(pretrained=True).to(device)` and assert `model.sample_rate == 16000, model.chin == 1, model.chout == 1, model.hidden == 48`.
2. **Forward Pass on Real 16 kHz Audio:** Load one real resampled 16 kHz noisy/clean .wav pair ([1, 1, T]), run `out = model(noisy)`, and assert `out.dtype == torch.float32`.
3. **Output Length & Shape:** Assert `out.shape == noisy.shape == clean.shape` for both an arbitrary full-length utterance ([1, 1, T]) and a 4-second training segment ([2, 1, 64000]).
4. **L1 + STFT Loss Calculation:** Compute `l1 = F.l1_loss(clean, out)` and `sc_loss, mag_loss = mr_stft_loss(out.squeeze(1), clean.squeeze(1))`, combine as `loss = l1 + 0.1 * sc_loss + 0.1 * mag_loss`, and assert `loss.ndim == 0` and `loss.item() > 0`.
5. **Backward Pass & Optimizer Update:** Snapshot one parameter tensor (`p_before = next(model.parameters()).clone()`), run `optimizer.zero_grad(), loss.backward()`, assert all parameter gradients are finite (`torch.isfinite(p.grad).all()`), run `optimizer.step()`, and assert `not torch.equal(p_before, next(model.parameters()))`.
6. **Checkpoint Save & Reload:** Save `model.state_dict()` and `optimizer.state_dict()` to a temporary .th file, instantiate a fresh Demucs instance, load the saved state dict, and assert `torch.allclose(model(noisy), reloaded_model(noisy), atol=1e-6)`.
7. **No NaN / Inf Values:** Explicitly assert `torch.isfinite(out).all()` and `torch.isfinite(loss).all()`, and verify `pesq(16000, clean_np, out_np, 'wb')` and `stoi(clean_np, out_np, 16000, extended=False)` return finite floats.

### D. Data Integrity & Strict Evaluation Isolation

- **Data Integrity Checks:** (1) Exact 1-to-1 filename set equality between clean and noisy directories (`set(clean_files) == set(noisy_files)`). (2) Exact sample rate (16000 Hz), channel count (1), and sample count equality (`len(clean_wav) == len(noisy_wav)`) verified for every single pair during preprocessing and manifest creation. (3) Pairwise disjoint speaker sets enforced via `assert set(tr_spk).isdisjoint(cv_spk) and set(tr_spk).isdisjoint(tt_spk) and set(cv_spk).isdisjoint(tt_spk)`.
- **Strict Test Set Isolation:** The 824 official test utterances (p232, p257 in manifests/tt/) are NEVER passed to 06_finetune_demucs.py. Early stopping, learning rate adjustments, and best_val_pesq.th / best_val_loss.th selection depend exclusively on the 2-speaker validation split (manifests/cv/). The test split (manifests/tt/) is read only by the read-only evaluation scripts (05_evaluate_baselines.py and 07_evaluate_and_report.py) under model.eval() and torch.no_grad().

## 2. Required Corrections to Previous Plan

1. **Do Not Use denoiser/audio.py (Audioset) or denoiser/solver.py:** Because denoiser/audio.py calls the removed `torchaudio.info().si.length` API and denoiser/solver.py is tightly coupled to legacy Hydra and denoiser/stft_loss.py, our `src/data/dataset.py` and `src/training/trainer.py` must be standalone PyTorch 2.x modules using `soundfile` for audio metadata and loading.
2. **Numerical Stability Floor in PyTorch 2.x STFT Loss:** When computing magnitude from complex STFT tensors via `torch.abs(torch.stft(..., return_complex=True))`, exact zeros in silent frames produce NaN gradients during backward() (since d|x|/dx is undefined at 0). Our `src/training/stft_loss.py` must compute `mag = torch.sqrt(x_stft.real**2 + x_stft.imag**2 + 1e-7)` (matching the original repo epsilon floor inside the square root) so gradients remain finite on silent regions.
3. **PyTorch 2.4+ torch.load Security Default (weights_only):** Modern PyTorch versions emit warnings or errors when loading checkpoints containing numpy scalars or dicts without `weights_only=False` (or safe globals). Our checkpoint loader and pretrained.dns48() wrapper will handle torch.load safely and store pure Python floats in checkpoint history.

## 3. Remaining Technical Risks

1. **RNN / CuDNN Backward Pass Requirement in eval() vs. train():** CuDNN LSTM backward passes require the module to be in `model.train()` mode; calling `.backward()` while `model.eval()` is active raises `RuntimeError: cudnn RNN backward can only be called in training mode`. Our trainer strictly separates `model.train()` (training step) from `model.eval() + torch.no_grad()` (validation/test step).
2. **Google Drive Partial Write on Disconnect:** If a Colab VM terminates mid-write while saving `checkpoint.th` to the Google Drive FUSE mount, the checkpoint file can be corrupted. Mitigation: write to `checkpoint.th.tmp` first and call `os.replace('checkpoint.th.tmp', 'checkpoint.th')`.

## 4. Exact First Implementation Step (Upon Approval)

Once you approve this final technical review, our exact first implementation step will be:

1. Create the core project configuration (`configs/config.yaml, requirements.txt`) and the minimal model/loss/metric modules (`src/utils/seed.py, src/utils/audio_io.py, src/models/demucs_loader.py, src/training/stft_loss.py`), followed immediately by `scripts/04_smoke_test.py` so we can execute and verify all 7 smoke-test checks before building the dataset download, preprocessing, baseline, and training scripts.
