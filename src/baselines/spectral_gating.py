"""Stationary Spectral Gating Baseline using noisereduce."""

from typing import Union
import noisereduce as nr
import numpy as np
import torch


class SpectralGatingBaseline:
    """Traditional speech enhancement baseline using stationary spectral gating."""

    def __init__(
        self,
        sr: int = 16000,
        stationary: bool = True,
        n_fft: int = 512,
        win_length: int = 512,
        hop_length: int = 128,
        n_std_thresh_stationary: float = 1.5,
        prop_decrease: float = 1.0,
    ):
        self.sr = sr
        self.stationary = stationary
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_std_thresh_stationary = n_std_thresh_stationary
        self.prop_decrease = prop_decrease

    def _enhance_1d_numpy(self, y: np.ndarray) -> np.ndarray:
        """Processes a single 1D float32 audio array [T]."""
        orig_len = len(y)
        if orig_len == 0:
            return y.copy()

        # Handle silence or near-zero variance
        variance = float(np.var(y))
        if variance < 1e-10 or not np.isfinite(y).all():
            out = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
            return np.clip(out, -1.0, 1.0)

        try:
            reduced = nr.reduce_noise(
                y=y,
                sr=self.sr,
                stationary=self.stationary,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_std_thresh_stationary=self.n_std_thresh_stationary,
                prop_decrease=self.prop_decrease,
            )
        except Exception:
            # Fallback safely to input if noisereduce fails numerically
            reduced = y.copy()

        # Handle non-finite values if generated
        if not np.isfinite(reduced).all():
            reduced = np.nan_to_num(reduced, nan=0.0, posinf=1.0, neginf=-1.0)

        # Enforce exact length parity
        if len(reduced) < orig_len:
            reduced = np.pad(reduced, (0, orig_len - len(reduced)))
        elif len(reduced) > orig_len:
            reduced = reduced[:orig_len]

        return np.clip(reduced, -1.0, 1.0).astype(np.float32)

    def enhance(
        self, audio: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Enhances input audio array or tensor and returns identical shape, type, and length."""
        is_torch = isinstance(audio, torch.Tensor)
        if is_torch:
            device = audio.device
            dtype = audio.dtype
            arr = audio.detach().cpu().float().numpy()
        else:
            arr = np.asarray(audio, dtype=np.float32)

        orig_shape = arr.shape

        if arr.ndim == 1:
            out_arr = self._enhance_1d_numpy(arr)
        elif arr.ndim == 2:
            # Shape [C, T]
            channels = [self._enhance_1d_numpy(arr[c]) for c in range(arr.shape[0])]
            out_arr = np.stack(channels, axis=0)
        elif arr.ndim == 3:
            # Shape [B, C, T]
            batches = []
            for b in range(arr.shape[0]):
                channels = [self._enhance_1d_numpy(arr[b, c]) for c in range(arr.shape[1])]
                batches.append(np.stack(channels, axis=0))
            out_arr = np.stack(batches, axis=0)
        else:
            raise ValueError(f"Unsupported audio shape {orig_shape}. Expected 1D, 2D, or 3D.")

        assert out_arr.shape == orig_shape, f"Shape mismatch: {out_arr.shape} != {orig_shape}"

        if is_torch:
            return torch.from_numpy(out_arr).to(device=device, dtype=dtype)
        return out_arr

    def __call__(
        self, audio: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self.enhance(audio)


def enhance_spectral_gating(
    audio: Union[np.ndarray, torch.Tensor],
    sr: int = 16000,
    **kwargs,
) -> Union[np.ndarray, torch.Tensor]:
    """Functional interface for stationary spectral gating."""
    baseline = SpectralGatingBaseline(sr=sr, **kwargs)
    return baseline.enhance(audio)
