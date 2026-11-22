"""Step 1 Smoke Test for Speech-Enhancement (Google Colab / PyTorch 2.x verification).

Usage:
  1. Stage 1 — Synthetic Smoke Test (No dataset download required — run first):
     python scripts/04_smoke_test.py

     Verifies ONLY:
       - Official pretrained Demucs dns48 loading & attribute verification
       - Inference forward pass (model.eval() + torch.no_grad()) & output shape/length parity
       - Training mode (model.train() called BEFORE training forward pass)
       - Training forward pass on 4.0s 16 kHz batch ([2, 1, 64000])
       - L1 + Multi-Resolution STFT loss (0.1 weights applied once in caller)
       - Backward pass, finite gradients on all parameters, and optimizer.step() parameter update
       - Atomic checkpoint save and reload equivalence
       - Strict absence of NaN or Inf values across all outputs, losses, and gradients

  2. Stage 2 — Real-Audio Pair Smoke Test (Run after dataset license verification & 1 real sample pair):
     python scripts/04_smoke_test.py --clean_wav path/to/clean.wav --noisy_wav path/to/noisy.wav

     Verifies:
       - Real WAV loading, sample-rate/alignment verification (and 48k->16k resampling if needed)
       - Pretrained dns48 inference on real speech
       - Wideband PESQ (16 kHz 'wb') and STOI (16 kHz) calculation on real clean/noisy/enhanced speech
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import yaml

# Add repository root to sys.path so `src.*` imports resolve cleanly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.demucs_loader import (
    build_demucs_dns48_architecture,
    load_checkpoint,
    load_pretrained_dns48,
    save_checkpoint_atomic,
    verify_demucs_dns48_attributes,
)
from src.training.stft_loss import MultiResolutionSTFTLoss
from src.utils.audio_io import (
    load_wav,
    resample_wav_tensor,
    verify_aligned_pair,
)
from src.utils.seed import set_global_seed


def generate_synthetic_16k_pair(
    sample_rate: int = 16000,
    duration_sec: float = 2.25,
    batch_size: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a deterministic synthetic 16 kHz waveform pair for tensor/gradient verification."""
    num_samples = int(sample_rate * duration_sec)
    t = torch.linspace(0.0, duration_sec, num_samples, device=device, dtype=torch.float32)

    envelope = 0.5 * (1.0 - torch.cos(2.0 * math.pi * 2.0 * t))
    clean_1d = 0.35 * envelope * (
        torch.sin(2.0 * math.pi * 220.0 * t)
        + 0.5 * torch.sin(2.0 * math.pi * 440.0 * t)
        + 0.25 * torch.sin(2.0 * math.pi * 880.0 * t)
    )

    clean = clean_1d.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
    noise = 0.08 * torch.randn_like(clean)
    noisy = torch.clamp(clean + noise, -1.0, 1.0)
    return clean, noisy


def run_synthetic_smoke_test(config: dict, device: torch.device) -> None:
    """Runs Stage 1 synthetic 16 kHz checks (no PESQ/STOI on synthetic audio)."""
    print("=" * 72)
    print("[STAGE 1] SYNTHETIC 16 kHz SMOKE TEST (No dataset download required)")
    print("=" * 72)

    sr = int(config["dataset"]["sample_rate"])
    sc_weight = float(config["loss"]["stft_sc_factor"])
    mag_weight = float(config["loss"]["stft_mag_factor"])
    assert sc_weight == 0.1 and mag_weight == 0.1, (
        f"Expected stft_sc_factor=0.1 and stft_mag_factor=0.1, got {sc_weight}, {mag_weight}"
    )

    # 1. Model loading
    print("[1/7] Loading official pretrained Demucs dns48 checkpoint...")
    model = load_pretrained_dns48(device=device)
    verify_demucs_dns48_attributes(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"      PASS: Loaded dns48 on {device} ({num_params:,} parameters, SR={model.sample_rate} Hz).")

    # 2. Separate Inference Test: model.eval() and torch.no_grad()
    print("[2/7] Inference check: model.eval() + torch.no_grad() forward pass & shape/length parity...")
    clean_eval, noisy_eval = generate_synthetic_16k_pair(
        sample_rate=sr, duration_sec=2.37, batch_size=2, device=device
    )
    model.eval()
    assert not model.training, "Model must be in eval() mode for the inference test!"
    with torch.no_grad():
        est_eval = model(noisy_eval)

    assert est_eval.dtype == torch.float32, f"Expected float32, got {est_eval.dtype}"
    assert est_eval.shape == noisy_eval.shape == clean_eval.shape, (
        f"Shape mismatch: est={est_eval.shape}, noisy={noisy_eval.shape}, clean={clean_eval.shape}"
    )
    assert torch.isfinite(est_eval).all(), "NaN or Inf detected in eval() output!"
    print(f"      PASS: Inference output shape {tuple(est_eval.shape)} matches input length; all values finite.")

    # 3. Training Mode & Training Forward Pass: call model.train() BEFORE forward pass
    print("[3/7] Training mode check: calling model.train() BEFORE new training forward pass...")
    clean_train, noisy_train = generate_synthetic_16k_pair(
        sample_rate=sr, duration_sec=4.0, batch_size=2, device=device
    )
    assert noisy_train.shape == (2, 1, 64000), f"Expected (2, 1, 64000), got {tuple(noisy_train.shape)}"

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        betas=tuple(config["training"]["betas"]),
    )
    optimizer.zero_grad()

    # Snapshot first parameter tensor before training step
    first_param_name, first_param = next(iter(model.named_parameters()))
    param_before = first_param.detach().clone()

    # Explicitly enter training mode BEFORE the training forward pass
    model.train()
    assert model.training, "Model must be in model.train() mode before the training forward pass!"

    est_train = model(noisy_train)
    assert est_train.requires_grad, "Training forward pass output must track gradients!"
    assert est_train.shape == clean_train.shape == (2, 1, 64000)
    assert torch.isfinite(est_train).all(), "NaN or Inf detected in train() forward output!"
    print(f"      PASS: Training forward pass completed in model.train() mode with shape {tuple(est_train.shape)}.")

    # 4. L1 + MR-STFT Loss Calculation (0.1 weights applied once in caller)
    print("[4/7] Loss check: computing L1 + 0.1 * raw_sc_loss + 0.1 * raw_mag_loss...")
    mr_stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=config["loss"]["fft_sizes"],
        hop_sizes=config["loss"]["hop_sizes"],
        win_lengths=config["loss"]["win_lengths"],
    ).to(device)

    l1_loss = F.l1_loss(est_train, clean_train)
    raw_sc_loss, raw_mag_loss = mr_stft_loss(est_train, clean_train)

    assert torch.isfinite(l1_loss) and l1_loss.ndim == 0 and l1_loss.item() > 0.0
    assert torch.isfinite(raw_sc_loss) and raw_sc_loss.ndim == 0 and raw_sc_loss.item() > 0.0
    assert torch.isfinite(raw_mag_loss) and raw_mag_loss.ndim == 0 and raw_mag_loss.item() > 0.0

    # Apply 0.1 weights exactly ONCE here
    total_loss = l1_loss + sc_weight * raw_sc_loss + mag_weight * raw_mag_loss
    assert torch.isfinite(total_loss) and total_loss.ndim == 0 and total_loss.item() > 0.0
    print(
        f"      PASS: l1={l1_loss.item():.5f}, raw_sc={raw_sc_loss.item():.5f}, "
        f"raw_mag={raw_mag_loss.item():.5f}, total_loss={total_loss.item():.5f} (all finite)."
    )

    # 5. Backward Pass & Finite Gradients Check
    print("[5/7] Backward check: running total_loss.backward() and verifying finite gradients...")
    total_loss.backward()

    grad_count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing gradient for parameter: {name}"
            assert torch.isfinite(p.grad).all(), f"NaN or Inf gradient in parameter: {name}"
            grad_count += 1
    print(f"      PASS: Backward pass succeeded; all {grad_count} parameter tensors have finite gradients.")

    # 6. Optimizer Step & Parameter Update Verification
    print("[6/7] Optimizer check: clipping gradients and running optimizer.step()...")
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float(config["training"]["grad_clip_max_norm"])
    )
    optimizer.step()

    param_after = first_param.detach().clone()
    assert torch.isfinite(param_after).all(), "NaN or Inf in updated model parameters!"
    assert not torch.equal(param_before, param_after), (
        f"Parameter '{first_param_name}' did not update after optimizer.step()!"
    )
    print(f"      PASS: Optimizer updated model parameters cleanly (no NaN/Inf).")

    # 7. Checkpoint Save & Reload Equivalence
    print("[7/7] Checkpoint check: atomic save and reload equivalence in eval() mode...")
    model.eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "smoke_test_checkpoint.th"
        save_checkpoint_atomic(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=1,
            history=[{"epoch": 1, "train_loss": float(total_loss.item())}],
        )
        assert ckpt_path.exists(), "Checkpoint file was not created!"

        reloaded_model = build_demucs_dns48_architecture().to(device)
        reloaded_opt = torch.optim.Adam(reloaded_model.parameters(), lr=float(config["training"]["lr"]))
        payload = load_checkpoint(
            checkpoint_path=ckpt_path,
            model=reloaded_model,
            optimizer=reloaded_opt,
            device=device,
        )
        assert payload["epoch"] == 1
        reloaded_model.eval()

        with torch.no_grad():
            out_orig = model(noisy_eval)
            out_reloaded = reloaded_model(noisy_eval)

        assert torch.isfinite(out_orig).all() and torch.isfinite(out_reloaded).all()
        max_diff = torch.max(torch.abs(out_orig - out_reloaded)).item()
        assert torch.allclose(out_orig, out_reloaded, atol=1e-6), (
            f"Reloaded model output diverged! Max diff = {max_diff}"
        )
    print(f"      PASS: Reloaded checkpoint matches original model (max_abs_diff={max_diff:.2e}).")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"      GPU Peak Allocated Memory during Stage 1 smoke test: {peak_mb:.1f} MB")


def run_real_pair_smoke_test(
    clean_wav_path: Path,
    noisy_wav_path: Path,
    config: dict,
    device: torch.device,
) -> None:
    """Runs Stage 2 real-audio verification including PESQ (16 kHz 'wb') and STOI."""
    print("\n" + "=" * 72)
    print("[STAGE 2] REAL AUDIO PAIR & METRICS (PESQ / STOI) SMOKE TEST")
    print("=" * 72)

    from pesq import pesq
    from pystoi import stoi

    target_sr = int(config["dataset"]["sample_rate"])
    clean_wav, clean_sr = load_wav(clean_wav_path, expected_sr=None, mono=True)
    noisy_wav, noisy_sr = load_wav(noisy_wav_path, expected_sr=None, mono=True)

    assert clean_sr == noisy_sr, f"Sample rate mismatch: clean={clean_sr}, noisy={noisy_sr}"
    assert clean_wav.shape == noisy_wav.shape, (
        f"Raw shape mismatch: clean={clean_wav.shape}, noisy={noisy_wav.shape}"
    )

    if clean_sr != target_sr:
        print(f"Resampling real pair from {clean_sr} Hz -> {target_sr} Hz...")
        clean_wav = resample_wav_tensor(clean_wav, orig_sr=clean_sr, target_sr=target_sr)
        noisy_wav = resample_wav_tensor(noisy_wav, orig_sr=noisy_sr, target_sr=target_sr)

    verify_aligned_pair(clean_wav, noisy_wav, target_sr, target_sr, expected_sr=target_sr)

    model = load_pretrained_dns48(device=device)
    model.eval()

    noisy_batch = noisy_wav.unsqueeze(0).to(device)  # [1, 1, T]
    clean_batch = clean_wav.unsqueeze(0).to(device)  # [1, 1, T]

    with torch.no_grad():
        enhanced_batch = model(noisy_batch)

    assert enhanced_batch.shape == noisy_batch.shape == clean_batch.shape
    assert torch.isfinite(enhanced_batch).all(), "Non-finite values in real audio inference!"

    clean_np = clean_batch[0, 0].cpu().numpy()
    noisy_np = noisy_batch[0, 0].cpu().numpy()
    enh_np = enhanced_batch[0, 0].cpu().numpy()

    noisy_pesq = float(pesq(target_sr, clean_np, noisy_np, "wb"))
    enh_pesq = float(pesq(target_sr, clean_np, enh_np, "wb"))
    noisy_stoi = float(stoi(clean_np, noisy_np, target_sr, extended=False))
    enh_stoi = float(stoi(clean_np, enh_np, target_sr, extended=False))

    assert math.isfinite(noisy_pesq) and math.isfinite(enh_pesq)
    assert math.isfinite(noisy_stoi) and math.isfinite(enh_stoi)

    print(f"Real Pair ({clean_wav_path.name}, N={clean_np.shape[0]} samples @ {target_sr} Hz):")
    print(f"  Noisy      -> PESQ(wb): {noisy_pesq:.4f} | STOI: {noisy_stoi:.4f}")
    print(f"  dns48 Zero -> PESQ(wb): {enh_pesq:.4f} | STOI: {enh_stoi:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Speech-Enhancement Step 1 Smoke Test.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--clean_wav",
        type=str,
        default=None,
        help="Optional path to a single real clean WAV file (for Stage 2 PESQ/STOI check)",
    )
    parser.add_argument(
        "--noisy_wav",
        type=str,
        default=None,
        help="Optional path to a single real noisy WAV file (for Stage 2 PESQ/STOI check)",
    )
    args = parser.parse_args()
    if bool(args.clean_wav) != bool(args.noisy_wav):
        parser.error(
            "--clean_wav and --noisy_wav must be provided together"
        )

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_global_seed(int(config["project"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Python version : {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device         : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Stage 1: Synthetic 16 kHz smoke test (no PESQ/STOI calculation on synthetic signals)
    run_synthetic_smoke_test(config=config, device=device)

    # Stage 2: Real clean/noisy speech pair check + PESQ/STOI evaluation (when paths are provided)
    if args.clean_wav and args.noisy_wav:
        run_real_pair_smoke_test(
            clean_wav_path=Path(args.clean_wav),
            noisy_wav_path=Path(args.noisy_wav),
            config=config,
            device=device,
        )
    else:
        print(
            "\n[INFO] Stage 2 (real-audio PESQ/STOI check) skipped because --clean_wav / --noisy_wav "
            "were not provided. Run with a real clean/noisy speech pair after license verification."
        )

    print("\nALL SMOKE TEST CHECKS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
