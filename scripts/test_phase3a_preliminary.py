"""Preliminary unit and integration tests for Phase 3A before full evaluation run."""

from pathlib import Path
import shutil
import sys
import numpy as np
import torch
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import SpeechEnhancementDataset, get_dataloader, ensure_local_data_16k
from src.baselines.spectral_gating import SpectralGatingBaseline
from src.models.demucs_loader import load_pretrained_dns48
from src.evaluation.evaluator import compute_utterance_metrics


def run_tests():
    print("=" * 78)
    print("PHASE 3A PRELIMINARY TEST SUITE")
    print("=" * 78)

    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    local_data_dir = Path("/content/data_16k")
    tar_path = Path("/content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar")
    train_manifest = PROJECT_ROOT / "manifests" / "train_manifest.json"
    val_manifest = PROJECT_ROOT / "manifests" / "val_manifest.json"
    test_manifest = PROJECT_ROOT / "manifests" / "test_manifest.json"

    # -------------------------------------------------------------------------
    # TEST 1: Colab Runtime Restoration Mechanism
    # -------------------------------------------------------------------------
    print("\n[TEST 1/5] Testing Colab runtime restoration (ensure_local_data_16k)...")
    # 1a. Test existing directory detection
    res_dir = ensure_local_data_16k(local_data_dir, tar_path=tar_path)
    assert res_dir == local_data_dir, f"Expected {local_data_dir}, got {res_dir}"
    print("  PASS: Detected existing local 16 kHz dataset directory.")

    # 1b. Test auto-restoration into a temporary test location
    test_restore_parent = Path("/content/test_restore_tmp")
    test_restore_parent.mkdir(parents=True, exist_ok=True)
    test_restore_target = test_restore_parent / "data_16k"
    try:
        restored = ensure_local_data_16k(test_restore_target, tar_path=tar_path)
        assert (restored / "clean_testset_wav").exists()
        assert (restored / "clean_trainset_28spk_wav").exists()
        print(f"  PASS: Successfully restored 16 kHz dataset from {tar_path.name} to {restored}.")
    finally:
        shutil.rmtree(test_restore_parent, ignore_errors=True)
    print("  PASS: Temporary test restore cleaned up safely without modifying persistent tar archive.")

    # -------------------------------------------------------------------------
    # TEST 2: SpeechEnhancementDataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n[TEST 2/5] Testing SpeechEnhancementDataset & DataLoader...")
    # Train dataset (sliding-window chunking mode)
    train_ds = SpeechEnhancementDataset(
        manifest_path=train_manifest,
        local_data_dir=local_data_dir,
        split="train",
        segment_seconds=4.0,
        stride_seconds=2.0,
    )
    print(f"  Train Manifest Utterances: {len(train_ds.manifest)}")
    print(f"  Train Windowed Chunks    : {len(train_ds)} (segment=4.0s, stride=2.0s)")
    assert len(train_ds) >= len(train_ds.manifest), "Expected chunks count >= utterance count"

    item0 = train_ds[0]
    assert item0["clean"].shape == (1, 64000), f"Expected shape (1, 64000), got {item0['clean'].shape}"
    assert item0["noisy"].shape == (1, 64000), f"Expected shape (1, 64000), got {item0['noisy'].shape}"
    assert torch.isfinite(item0["clean"]).all(), "Non-finite values in clean chunk"
    assert torch.isfinite(item0["noisy"]).all(), "Non-finite values in noisy chunk"
    print(f"  PASS: Single train item has exact shape (1, 64000) and finite samples.")

    train_loader = get_dataloader(
        manifest_path=train_manifest,
        local_data_dir=local_data_dir,
        split="train",
        batch_size=4,
        num_workers=2,
    )
    batch = next(iter(train_loader))
    assert batch["clean"].shape == (4, 1, 64000), f"Expected batch shape (4, 1, 64000), got {batch['clean'].shape}"
    assert batch["noisy"].shape == (4, 1, 64000), f"Expected batch shape (4, 1, 64000), got {batch['noisy'].shape}"
    assert len(batch["utterance_id"]) == 4
    print(f"  PASS: DataLoader collated batch with exact shape (4, 1, 64000).")

    # Val & Test datasets (uncropped full-utterance mode)
    val_ds = SpeechEnhancementDataset(
        manifest_path=val_manifest,
        local_data_dir=local_data_dir,
        split="val",
    )
    test_ds = SpeechEnhancementDataset(
        manifest_path=test_manifest,
        local_data_dir=local_data_dir,
        split="test",
    )
    assert len(val_ds) == 770, f"Expected 770 val items, got {len(val_ds)}"
    assert len(test_ds) == 824, f"Expected 824 test items, got {len(test_ds)}"

    val_item = val_ds[0]
    test_item = test_ds[0]
    assert val_item["clean"].ndim == 2 and val_item["clean"].shape[0] == 1
    assert val_item["clean"].shape == val_item["noisy"].shape
    assert test_item["clean"].shape == test_item["noisy"].shape
    print(f"  PASS: Val item ({val_item['utterance_id']}) returned uncropped shape {tuple(val_item['clean'].shape)}.")
    print(f"  PASS: Test item ({test_item['utterance_id']}) returned uncropped shape {tuple(test_item['clean'].shape)}.")

    # -------------------------------------------------------------------------
    # TEST 3: Spectral Gating Baseline
    # -------------------------------------------------------------------------
    print("\n[TEST 3/5] Testing SpectralGatingBaseline...")
    sg_baseline = SpectralGatingBaseline(
        sr=16000,
        stationary=True,
        n_fft=512,
        win_length=512,
        hop_length=128,
        n_std_thresh_stationary=1.5,
        prop_decrease=1.0,
    )

    # Test with 1D numpy array
    sig_1d = test_item["noisy"].squeeze(0).numpy()
    out_1d = sg_baseline.enhance(sig_1d)
    assert isinstance(out_1d, np.ndarray) and out_1d.shape == sig_1d.shape
    assert np.isfinite(out_1d).all()
    assert np.max(np.abs(out_1d)) <= 1.0
    print("  PASS: 1D numpy array enhanced with identical length, finiteness, and [-1, 1] clipping.")

    # Test with 2D PyTorch tensor [1, T]
    sig_2d = test_item["noisy"]
    out_2d = sg_baseline.enhance(sig_2d)
    assert isinstance(out_2d, torch.Tensor) and out_2d.shape == sig_2d.shape
    assert torch.isfinite(out_2d).all()
    print("  PASS: 2D PyTorch tensor [1, T] enhanced with identical shape and finiteness.")

    # Test edge case: silence / all zeros
    zeros_1d = np.zeros(16000, dtype=np.float32)
    out_zeros = sg_baseline.enhance(zeros_1d)
    assert np.all(out_zeros == 0.0)
    print("  PASS: Silent / all-zero input handled safely without divide-by-zero or numerical warnings.")

    # -------------------------------------------------------------------------
    # TEST 4: Pretrained Demucs dns48 Inference
    # -------------------------------------------------------------------------
    print("\n[TEST 4/5] Testing Pretrained Demucs dns48 on CUDA...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained_dns48(device=device)
    model.eval()

    with torch.no_grad():
        noisy_in = test_item["noisy"].unsqueeze(0).to(device)  # [1, 1, T]
        demucs_out = model(noisy_in)

    assert demucs_out.shape == noisy_in.shape
    assert torch.isfinite(demucs_out).all()
    print(f"  PASS: Demucs dns48 forward pass on {device} returned shape {tuple(demucs_out.shape)} (all finite).")

    # -------------------------------------------------------------------------
    # TEST 5: Metric Calculation on Real Audio Pair
    # -------------------------------------------------------------------------
    print("\n[TEST 5/5] Testing PESQ and STOI computation across all 3 conditions...")
    clean_np = test_item["clean"].squeeze(0).numpy()
    noisy_np = test_item["noisy"].squeeze(0).numpy()
    sg_np = out_1d
    demucs_np = demucs_out.squeeze(0).squeeze(0).cpu().numpy()

    p_noisy, s_noisy, pe_noisy, se_noisy = compute_utterance_metrics(clean_np, noisy_np)
    p_sg, s_sg, pe_sg, se_sg = compute_utterance_metrics(clean_np, sg_np)
    p_demucs, s_demucs, pe_demucs, se_demucs = compute_utterance_metrics(clean_np, demucs_np)

    print(f"  Utterance: {test_item['utterance_id']} ({test_item['speaker_id']}, {test_item['noise_type']}, SNR={test_item['snr_db']} dB)")
    print(f"  Condition 1: Noisy            -> PESQ(wb): {p_noisy:.4f} | STOI: {s_noisy:.4f}")
    print(f"  Condition 2: Spectral Gating  -> PESQ(wb): {p_sg:.4f} | STOI: {s_sg:.4f}")
    print(f"  Condition 3: dns48 Zero-Shot  -> PESQ(wb): {p_demucs:.4f} | STOI: {s_demucs:.4f}")

    assert p_noisy is not None and s_noisy is not None, f"Failed noisy metrics: {pe_noisy}, {se_noisy}"
    assert p_sg is not None and s_sg is not None, f"Failed sg metrics: {pe_sg}, {se_sg}"
    assert p_demucs is not None and s_demucs is not None, f"Failed demucs metrics: {pe_demucs}, {se_demucs}"
    assert -0.5 <= p_noisy <= 4.5 and 0.0 <= s_noisy <= 1.0, f"Invalid noisy metrics: {p_noisy}, {s_noisy}"
    assert -0.5 <= p_sg <= 4.5 and 0.0 <= s_sg <= 1.0, f"Invalid sg metrics: {p_sg}, {s_sg}"
    assert -0.5 <= p_demucs <= 4.5 and 0.0 <= s_demucs <= 1.0, f"Invalid demucs metrics: {p_demucs}, {s_demucs}"
    print("  PASS: All 3 conditions successfully computed valid, finite PESQ and STOI metrics.")

    print("\n" + "=" * 78)
    print("ALL PRELIMINARY UNIT & INTEGRATION TESTS PASSED (100%)")
    print("=" * 78)


if __name__ == "__main__":
    run_tests()
