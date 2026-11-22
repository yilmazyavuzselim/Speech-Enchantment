"""Audio preprocessing: extraction, 48 kHz validation, 16 kHz resampling, and tar packaging."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import tarfile
import time
from typing import Dict, Any, List, Tuple, Optional
import soundfile as sf
import torch
import torchaudio
import zipfile

from src.utils.audio_io import load_wav, save_wav, resample_wav_tensor


def extract_zip(zip_path: Path, extract_dir: Path) -> List[Path]:
    """Extracts a zip file into extract_dir and returns list of extracted wav files."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_wavs = []
    print(f"Extracting {zip_path.name} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/") or "__MACOSX" in member:
                continue
            zf.extract(member, extract_dir)
            p = extract_dir / member
            if p.suffix.lower() == ".wav":
                extracted_wavs.append(p)
    print(f"  Extracted {len(extracted_wavs)} WAV files from {zip_path.name}")
    return extracted_wavs


def _resample_single_file_worker(args: Tuple[str, str, int, int]) -> Dict[str, Any]:
    """Worker function to validate 48k audio, resample to 16k, and save as 16-bit PCM."""
    src_path_str, dst_path_str, orig_sr, target_sr = args
    src_path = Path(src_path_str)
    dst_path = Path(dst_path_str)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data, sr = sf.read(str(src_path), dtype="float32", always_2d=True)
        # Verify 48k properties
        if sr != orig_sr:
            return {"file": src_path.name, "error": f"Sample rate mismatch: {sr} != {orig_sr}"}
        channels = data.shape[1]
        if channels != 1:
            return {"file": src_path.name, "error": f"Channel count mismatch: {channels} != 1 (mono expected)"}

        wav = torch.from_numpy(data.T)  # [1, num_samples]
        if not torch.isfinite(wav).all():
            return {"file": src_path.name, "error": "NaN or Inf values detected"}

        # Resample
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            wav_16k = resampler(wav)
        else:
            wav_16k = wav

        num_samples_16k = wav_16k.shape[-1]
        duration_sec = num_samples_16k / target_sr

        # Save as 16-bit PCM
        wav_np = wav_16k.squeeze(0).numpy()
        wav_np = wav_np.clip(-1.0, 1.0)
        sf.write(str(dst_path), wav_np, target_sr, subtype="PCM_16")

        return {
            "file": src_path.name,
            "orig_samples": data.shape[0],
            "resampled_samples": num_samples_16k,
            "duration_sec": duration_sec,
            "error": None,
        }
    except Exception as e:
        return {"file": src_path.name, "error": str(e)}


def resample_directory_parallel(
    src_dir: Path,
    dst_dir: Path,
    orig_sr: int = 48000,
    target_sr: int = 16000,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Resamples all WAV files in src_dir to dst_dir in parallel using ProcessPoolExecutor."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    all_src_files = [p for p in src_dir.glob("**/*.wav") if not p.name.startswith(".")]

    if not all_src_files:
        raise ValueError(f"No WAV files found in source directory: {src_dir}")

    print(f"Resampling {len(all_src_files)} files from {src_dir} to {dst_dir} ({orig_sr} -> {target_sr} Hz)...")
    tasks = []
    for src in all_src_files:
        # Maintain filename directly inside dst_dir
        dst = dst_dir / src.name
        tasks.append((str(src), str(dst), orig_sr, target_sr))

    num_workers = max_workers or min(os.cpu_count() or 4, 8)
    start_time = time.time()
    processed_count = 0
    errors = []
    stats = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_resample_single_file_worker, t): t for t in tasks}
        last_log = time.time()
        for fut in as_completed(futures):
            res = fut.result()
            processed_count += 1
            if res["error"] is not None:
                errors.append(res)
            else:
                stats.append(res)

            curr = time.time()
            if curr - last_log >= 10.0 or processed_count == len(tasks):
                elapsed = curr - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"  Processed {processed_count}/{len(tasks)} files ({rate:.1f} files/sec)")
                last_log = curr

    total_time = time.time() - start_time
    print(f"Finished resampling {processed_count} files in {total_time:.1f}s (Errors: {len(errors)})")

    if errors:
        raise RuntimeError(f"Encountered {len(errors)} errors during resampling: {errors[:5]}")

    return {
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "total_files": processed_count,
        "elapsed_sec": round(total_time, 2),
        "errors": errors,
    }


def create_tar_archive(source_dir: Path, output_tar: Path) -> Dict[str, Any]:
    """Packages source_dir into a tar archive for persistent storage."""
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    temp_tar = output_tar.with_suffix(".tar.tmp")

    print(f"Packaging {source_dir} into {output_tar}...")
    start_time = time.time()

    with tarfile.open(temp_tar, "w") as tar:
        tar.add(source_dir, arcname=source_dir.name)

    temp_tar.replace(output_tar)
    elapsed = time.time() - start_time
    size_bytes = output_tar.stat().st_size
    size_gb = size_bytes / (1024 ** 3)
    print(f"Created archive {output_tar.name} ({size_gb:.2f} GB) in {elapsed:.1f}s")

    return {
        "tar_path": str(output_tar),
        "size_bytes": size_bytes,
        "size_gb": round(size_gb, 3),
        "elapsed_sec": round(elapsed, 2),
    }


def resample_dataset_to_16k(
    raw_zips_dir: Path,
    local_extract_dir: Path,
    local_16k_dir: Path,
    output_tar: Optional[Path] = None,
    orig_sr: int = 48000,
    target_sr: int = 16000,
    clean_temp_extracted: bool = True,
) -> Dict[str, Any]:
    """End-to-end preprocessing pipeline: extracts 48k zips, resamples to 16k, and packages to tar."""
    raw_zips_dir = Path(raw_zips_dir)
    local_extract_dir = Path(local_extract_dir)
    local_16k_dir = Path(local_16k_dir)

    dataset_subsets = [
        ("clean_trainset_28spk_wav.zip", "clean_trainset_28spk_wav"),
        ("noisy_trainset_28spk_wav.zip", "noisy_trainset_28spk_wav"),
        ("clean_testset_wav.zip", "clean_testset_wav"),
        ("noisy_testset_wav.zip", "noisy_testset_wav"),
    ]

    results = {}
    for zip_name, subset_folder in dataset_subsets:
        zip_path = raw_zips_dir / zip_name
        if not zip_path.exists():
            raise FileNotFoundError(f"Required raw zip archive not found: {zip_path}")

        extracted_sub_dir = local_extract_dir / subset_folder
        extract_zip(zip_path, extracted_sub_dir)

        # Resample to local 16k directory
        resampled_sub_dir = local_16k_dir / subset_folder
        sub_res = resample_directory_parallel(
            src_dir=extracted_sub_dir,
            dst_dir=resampled_sub_dir,
            orig_sr=orig_sr,
            target_sr=target_sr,
        )
        results[subset_folder] = sub_res

    # Also extract logfiles and text transcript archives into local_16k_dir for self-contained dataset
    for text_zip in ["logfiles.zip", "trainset_28spk_txt.zip", "testset_txt.zip"]:
        text_zip_path = raw_zips_dir / text_zip
        if text_zip_path.exists():
            extract_zip(text_zip_path, local_16k_dir / text_zip_path.stem)

    # Copy license_text if present
    license_src = raw_zips_dir / "license_text"
    if license_src.exists():
        shutil.copy2(license_src, local_16k_dir / "license_text")

    tar_info = None
    if output_tar is not None:
        tar_info = create_tar_archive(local_16k_dir, Path(output_tar))

    if clean_temp_extracted and local_extract_dir.exists():
        print(f"Cleaning up temporary 48 kHz extracted directory: {local_extract_dir}...")
        shutil.rmtree(local_extract_dir, ignore_errors=True)
        print("Cleanup completed.")

    return {
        "status": "SUCCESS",
        "subsets": results,
        "tar_archive": tar_info,
        "local_16k_dir": str(local_16k_dir),
    }
