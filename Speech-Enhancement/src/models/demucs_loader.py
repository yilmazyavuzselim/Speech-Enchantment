"""Loader and verification utilities for Facebook Research Demucs (dns48)."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

DNS_48_URL = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns48-11decc9d8e3f0998.th"


def build_demucs_dns48_architecture() -> nn.Module:
    """Instantiates an uninitialized Demucs architecture matching dns48."""
    try:
        from denoiser.demucs import Demucs
    except ImportError as exc:
        raise ImportError(
            "Could not import 'denoiser.demucs'. Install facebookresearch/denoiser via:\n"
            "  pip install --no-deps git+https://github.com/facebookresearch/denoiser.git\n"
            "  pip install julius"
        ) from exc

    model = Demucs(
        chin=1,
        chout=1,
        hidden=48,
        depth=5,
        kernel_size=8,
        stride=4,
        causal=True,
        resample=4,
        growth=2,
        max_hidden=10_000,
        normalize=True,
        glu=True,
        rescale=0.1,
        floor=1e-3,
        sample_rate=16000,
    )
    return model


def verify_demucs_dns48_attributes(model: nn.Module) -> None:
    """Verifies that the loaded Demucs instance has the exact dns48 configuration."""
    expected = {
        "chin": 1,
        "chout": 1,
        "hidden": 48,
        "depth": 5,
        "kernel_size": 8,
        "stride": 4,
        "causal": True,
        "resample": 4,
        "normalize": True,
        "sample_rate": 16000,
    }
    for attr, expected_val in expected.items():
        actual_val = getattr(model, attr, None)
        if actual_val != expected_val:
            raise ValueError(
                f"Demucs attribute mismatch for '{attr}': expected {expected_val}, got {actual_val}"
            )


def load_pretrained_dns48(
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """Loads the official pretrained Demucs dns48 checkpoint (never master64)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    try:
        from denoiser import pretrained

        model = pretrained.dns48(pretrained=True)
    except Exception:
        # Fallback direct load via torch.hub if pretrained.dns48() hits torch.load weights_only issues
        model = build_demucs_dns48_architecture()
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                DNS_48_URL, map_location="cpu", check_hash=True, weights_only=False
            )
        except TypeError:
            state_dict = torch.hub.load_state_dict_from_url(
                DNS_48_URL, map_location="cpu", check_hash=True
            )
        model.load_state_dict(state_dict, strict=True)

    verify_demucs_dns48_attributes(model)
    model = model.to(device)
    return model


def save_checkpoint_atomic(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    history: Optional[ list ] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically saves a training/model checkpoint using a temporary file and os.replace."""
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")

    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "history": history if history is not None else [],
        "extra": extra if extra is not None else {},
    }
    torch.save(payload, str(tmp_path))
    os.replace(str(tmp_path), str(target))


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Loads a saved checkpoint into the provided model and optional optimizer."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    try:
        payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(checkpoint_path), map_location=device)

    if model is not None:
        state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return payload
