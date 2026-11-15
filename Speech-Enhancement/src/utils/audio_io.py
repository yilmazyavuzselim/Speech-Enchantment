"""Audio I/O, resampling, and sample-alignment verification utilities."""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torchaudio


def get_audio_info(path: Union[str, Path]) -> Tuple[int, int, int]:
    """Returns (sample_rate, num_frames, channels) using soundfile."""
    info = sf.info(str(path))
    return info.samplerate, info.frames, info.channels


def load_wav(
    path: Union[str, Path],
    expected_sr: int = 16000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Loads a WAV file as a float32 Tensor of shape [channels, num_samples].

    Verifies sample rate and mono channel count if requested.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # data has shape [num_frames, channels] -> transpose to [channels, num_frames]
    wav = torch.from_numpy(data.T.copy())

    if expected_sr is not None and sr != expected_sr:
        raise ValueError(
            f"Sample rate mismatch for {path}: expected {expected_sr} Hz, got {sr} Hz."
        )
    if mono and wav.shape[0] != 1:
        raise ValueError(
            f"Channel count mismatch for {path}: expected mono (1 channel), got {wav.shape[0]}."
        )
    if not torch.isfinite(wav).all():
        raise ValueError(f"Non-finite (NaN/Inf) audio samples detected in {path}.")

    return wav, sr


def save_wav(
    path: Union[str, Path],
    wav: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    clip: bool = True,
) -> None:
    """Saves a mono/multi-channel waveform as 16-bit PCM WAV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(wav, torch.Tensor):
        wav_np = wav.detach().cpu().float().numpy()
    else:
        wav_np = np.asarray(wav, dtype=np.float32)

    if wav_np.ndim == 3:
        if wav_np.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 when saving 3D tensor, got {wav_np.shape}")
        wav_np = wav_np[0]

    if wav_np.ndim == 2:
        # Convert [channels, num_samples] -> [num_samples, channels] for soundfile
        wav_np = wav_np.T

    if clip:
        wav_np = np.clip(wav_np, -1.0, 1.0)

    sf.write(str(out_path), wav_np, sample_rate, subtype="PCM_16")


def resample_wav_tensor(
    wav: torch.Tensor,
    orig_sr: int,
    target_sr: int = 16000,
) -> torch.Tensor:
    """Resamples a waveform tensor [..., T] from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return wav
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
    ).to(wav.device)
    return resampler(wav)


def verify_aligned_pair(
    clean_wav: torch.Tensor,
    noisy_wav: torch.Tensor,
    clean_sr: int,
    noisy_sr: int,
    expected_sr: int = 16000,
) -> None:
    """Asserts that clean and noisy tensors are mono, finite, at expected_sr, and sample-aligned."""
    if clean_sr != expected_sr or noisy_sr != expected_sr:
        raise ValueError(
            f"Sample rate mismatch: clean={clean_sr}, noisy={noisy_sr}, expected={expected_sr}"
        )
    if clean_wav.shape != noisy_wav.shape:
        raise ValueError(
            f"Shape mismatch between clean {tuple(clean_wav.shape)} and noisy {tuple(noisy_wav.shape)}"
        )
    if clean_wav.ndim != 2 or clean_wav.shape[0] != 1:
        raise ValueError(
            f"Expected shape [1, num_samples], got {tuple(clean_wav.shape)}"
        )
    if not torch.isfinite(clean_wav).all() or not torch.isfinite(noisy_wav).all():
        raise ValueError("Non-finite values found in clean/noisy audio pair.")
