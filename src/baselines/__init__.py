"""Baseline algorithms for speech enhancement."""

from src.baselines.spectral_gating import SpectralGatingBaseline, enhance_spectral_gating

__all__ = [
    "SpectralGatingBaseline",
    "enhance_spectral_gating",
]
