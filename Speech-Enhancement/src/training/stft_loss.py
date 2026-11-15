"""PyTorch 2.x compatible Multi-Resolution STFT Loss module.

Adapted from facebookresearch/denoiser (denoiser/stft_loss.py) and ParallelWaveGAN.
IMPORTANT:
  - Uses torch.stft(..., return_complex=True) for PyTorch 2.x compatibility.
  - Uses torch.sqrt(x_stft.real**2 + x_stft.imag**2 + 1e-7) to prevent NaN gradients on silent frames.
  - Returns RAW (unweighted) (sc_loss, mag_loss) averaged across resolutions.
  - The 0.1 weights (stft_sc_factor=0.1, stft_mag_factor=0.1) MUST be applied once in the caller/trainer:
      total_loss = l1_loss + 0.1 * sc_loss + 0.1 * mag_loss
"""

from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def stft_magnitude(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """Computes STFT magnitude using PyTorch 2.x return_complex=True with a 1e-7 stability floor."""
    if x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError(f"Expected mono [B, 1, T] or [B, T], got shape {tuple(x.shape)}")
        x = x.squeeze(1)
    elif x.ndim != 2:
        raise ValueError(f"Expected 2D [B, T] or 3D [B, 1, T] tensor, got {tuple(x.shape)}")

    window = window.to(device=x.device, dtype=x.dtype)
    x_stft = torch.stft(
        x,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    # Match denoiser/stft_loss.py numerical floor inside sqrt to avoid NaN gradients at 0
    real = x_stft.real
    imag = x_stft.imag
    return torch.sqrt(torch.clamp(real * real + imag * imag, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss module."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """Computes Frobenius norm ratio ||y_mag - x_mag||_F / ||y_mag||_F."""
        num = torch.norm(y_mag - x_mag, p="fro")
        den = torch.norm(y_mag, p="fro").clamp(min=1e-7)
        return num / den


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude L1 loss module."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """Computes L1 loss between log(y_mag) and log(x_mag)."""
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(nn.Module):
    """Single-resolution STFT loss module returning raw (sc_loss, mag_loss)."""

    def __init__(
        self,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
    ) -> None:
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns raw (sc_loss, mag_loss) where x is predicted and y is clean target."""
        x_mag = stft_magnitude(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft_magnitude(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss module returning raw (sc_loss, mag_loss).

    Do NOT multiply by 0.1 inside this module; apply stft_sc_factor and stft_mag_factor
    once in the training loop:
        total_loss = l1_loss + 0.1 * sc_loss + 0.1 * mag_loss
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (1024, 2048, 512),
        hop_sizes: Sequence[int] = (120, 240, 50),
        win_lengths: Sequence[int] = (600, 1200, 240),
    ) -> None:
        super().__init__()
        if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("fft_sizes, hop_sizes, and win_lengths must have the same length.")

        self.stft_losses = nn.ModuleList(
            [
                STFTLoss(fft_size=fs, shift_size=ss, win_length=wl)
                for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths)
            ]
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns unweighted mean (sc_loss, mag_loss) across all STFT resolutions."""
        sc_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        mag_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss = sc_loss + sc_l
            mag_loss = mag_loss + mag_l

        num_resolutions = len(self.stft_losses)
        sc_loss = sc_loss / num_resolutions
        mag_loss = mag_loss / num_resolutions
        return sc_loss, mag_loss
