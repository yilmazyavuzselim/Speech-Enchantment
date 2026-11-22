"""PyTorch Dataset and DataLoader for Speech Enhancement on Edinburgh VoiceBank-DEMAND."""

import json
import math
from pathlib import Path
import shutil
import tarfile
from typing import Dict, Any, List, Optional, Tuple, Union
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.utils.seed import seed_worker


def ensure_local_data_16k(
    local_data_dir: Union[str, Path] = "/content/data_16k",
    tar_path: Union[str, Path] = "/content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar",
) -> Path:
    """Verifies that the local 16 kHz data directory exists, restoring it from persistent tar if missing."""
    local_dir = Path(local_data_dir)
    clean_test = local_dir / "clean_testset_wav"
    clean_train = local_dir / "clean_trainset_28spk_wav"

    if clean_test.exists() and clean_train.exists():
        return local_dir

    tar_file = Path(tar_path)
    if not tar_file.exists():
        raise FileNotFoundError(
            f"Local 16 kHz data directory '{local_dir}' is incomplete and persistent tar archive "
            f"'{tar_file}' was not found. Please ensure dataset preparation has been executed."
        )

    print(f"Local 16 kHz data directory incomplete. Restoring from persistent archive {tar_file}...")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_file, "r") as tar:
        tar.extractall(path=local_dir.parent)

    if not (clean_test.exists() and clean_train.exists()):
        raise RuntimeError(
            f"Extracted {tar_file} into {local_dir.parent}, but expected directories "
            f"{clean_test} and {clean_train} were not found."
        )

    print(f"Successfully restored 16 kHz dataset to {local_dir}")
    return local_dir


class SpeechEnhancementDataset(Dataset):
    """Speech Enhancement Dataset supporting train (sliding-window chunking) and val/test (full-utterance)."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        local_data_dir: Union[str, Path] = "/content/data_16k",
        split: str = "train",
        sample_rate: int = 16000,
        segment_seconds: float = 4.0,
        stride_seconds: float = 2.0,
        pad: bool = True,
        auto_restore: bool = True,
        tar_path: Union[str, Path] = "/content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar",
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.local_data_dir = Path(local_data_dir)
        self.split = split.lower()
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.stride_seconds = stride_seconds
        self.pad = pad

        if auto_restore and str(self.local_data_dir).startswith("/content"):
            ensure_local_data_16k(self.local_data_dir, tar_path=tar_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest: List[Dict[str, Any]] = json.load(f)

        self.segment_samples = int(self.segment_seconds * self.sample_rate)
        self.stride_samples = int(self.stride_seconds * self.sample_rate)

        # Build chunk index for training mode
        self.chunks: List[Tuple[int, int, int]] = []
        if self.split == "train":
            for utt_idx, entry in enumerate(self.manifest):
                num_samples = entry.get("num_samples", 0)
                if num_samples > self.segment_samples:
                    num_chunks = max(
                        1,
                        math.ceil((num_samples - self.segment_samples) / self.stride_samples) + 1,
                    )
                else:
                    num_chunks = 1

                for chunk_idx in range(num_chunks):
                    start_sample = chunk_idx * self.stride_samples
                    self.chunks.append((utt_idx, chunk_idx, start_sample))

    def _resolve_path(self, rel_path: Optional[str], abs_path: Optional[str]) -> Path:
        """Resolves path preferentially via local_data_dir / rel_path, with fallback to abs_path."""
        if rel_path:
            candidate = self.local_data_dir / rel_path
            if candidate.exists():
                return candidate
        if abs_path:
            candidate = Path(abs_path)
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not locate audio file. rel_path='{rel_path}', abs_path='{abs_path}', "
            f"local_data_dir='{self.local_data_dir}'"
        )

    def _load_audio_pair(self, entry: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads clean and noisy audio as float32 tensors of shape [1, T]."""
        clean_file = self._resolve_path(entry.get("clean_rel_path"), entry.get("clean_path"))
        noisy_file = self._resolve_path(entry.get("noisy_rel_path"), entry.get("noisy_path"))

        clean_data, clean_sr = sf.read(str(clean_file), dtype="float32", always_2d=True)
        noisy_data, noisy_sr = sf.read(str(noisy_file), dtype="float32", always_2d=True)

        if clean_sr != self.sample_rate or noisy_sr != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch for {entry.get('utterance_id')}: clean={clean_sr}, "
                f"noisy={noisy_sr}, expected={self.sample_rate}"
            )

        clean = torch.from_numpy(clean_data.T.copy())  # [1, T]
        noisy = torch.from_numpy(noisy_data.T.copy())  # [1, T]

        if clean.shape != noisy.shape:
            raise ValueError(
                f"Shape mismatch for {entry.get('utterance_id')}: clean={clean.shape}, noisy={noisy.shape}"
            )

        if not torch.isfinite(clean).all() or not torch.isfinite(noisy).all():
            raise ValueError(f"Non-finite audio samples in utterance {entry.get('utterance_id')}")

        return clean, noisy

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.chunks)
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.split == "train":
            utt_idx, chunk_idx, start_sample = self.chunks[index]
            entry = self.manifest[utt_idx]
            clean, noisy = self._load_audio_pair(entry)

            end_sample = start_sample + self.segment_samples
            clean_chunk = clean[:, start_sample:end_sample]
            noisy_chunk = noisy[:, start_sample:end_sample]

            pad_len = self.segment_samples - clean_chunk.shape[-1]
            if pad_len > 0 and self.pad:
                clean_chunk = F.pad(clean_chunk, (0, pad_len))
                noisy_chunk = F.pad(noisy_chunk, (0, pad_len))

            return {
                "clean": clean_chunk,
                "noisy": noisy_chunk,
                "utterance_id": entry["utterance_id"],
                "speaker_id": entry["speaker_id"],
                "noise_type": entry.get("noise_type", "unknown"),
                "snr_db": float(entry.get("snr_db", 0.0)),
                "chunk_idx": chunk_idx,
                "num_samples": self.segment_samples,
                "duration_sec": self.segment_seconds,
            }
        else:
            entry = self.manifest[index]
            clean, noisy = self._load_audio_pair(entry)
            num_samples = clean.shape[-1]
            duration_sec = num_samples / self.sample_rate

            return {
                "clean": clean,
                "noisy": noisy,
                "utterance_id": entry["utterance_id"],
                "speaker_id": entry["speaker_id"],
                "noise_type": entry.get("noise_type", "unknown"),
                "snr_db": float(entry.get("snr_db", 0.0)),
                "num_samples": num_samples,
                "duration_sec": round(duration_sec, 4),
            }


def collate_train_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates fixed-length training segments into batched tensors [B, 1, T]."""
    clean = torch.stack([item["clean"] for item in batch], dim=0)
    noisy = torch.stack([item["noisy"] for item in batch], dim=0)
    snr_db = torch.tensor([item["snr_db"] for item in batch], dtype=torch.float32)

    return {
        "clean": clean,
        "noisy": noisy,
        "utterance_id": [item["utterance_id"] for item in batch],
        "speaker_id": [item["speaker_id"] for item in batch],
        "noise_type": [item["noise_type"] for item in batch],
        "snr_db": snr_db,
        "chunk_idx": [item.get("chunk_idx", 0) for item in batch],
    }


def collate_eval_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates evaluation items without padding (typically batch_size=1)."""
    if len(batch) == 1:
        return batch[0]
    return {
        "clean": [item["clean"] for item in batch],
        "noisy": [item["noisy"] for item in batch],
        "utterance_id": [item["utterance_id"] for item in batch],
        "speaker_id": [item["speaker_id"] for item in batch],
        "noise_type": [item["noise_type"] for item in batch],
        "snr_db": [item["snr_db"] for item in batch],
        "num_samples": [item["num_samples"] for item in batch],
        "duration_sec": [item["duration_sec"] for item in batch],
    }


def get_dataloader(
    manifest_path: Union[str, Path],
    local_data_dir: Union[str, Path] = "/content/data_16k",
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 2,
    shuffle: Optional[bool] = None,
    seed: int = 42,
    **dataset_kwargs,
) -> DataLoader:
    """Builds a deterministic PyTorch DataLoader for train, val, or test."""
    dataset = SpeechEnhancementDataset(
        manifest_path=manifest_path,
        local_data_dir=local_data_dir,
        split=split,
        **dataset_kwargs,
    )

    is_train = split.lower() == "train"
    if shuffle is None:
        shuffle = is_train

    collate_fn = collate_train_batch if is_train else collate_eval_batch

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )
