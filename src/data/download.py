"""Dataset download and archive integrity verification for Edinburgh VoiceBank-DEMAND."""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
import urllib.request
import zipfile

EDINBURGH_BASE_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/"

OFFICIAL_BITSTREAMS = [
    {
        "filename": "clean_trainset_28spk_wav.zip",
        "expected_bytes": 2486057279,
        "is_zip": True,
    },
    {
        "filename": "noisy_trainset_28spk_wav.zip",
        "expected_bytes": 2830205201,
        "is_zip": True,
    },
    {
        "filename": "clean_testset_wav.zip",
        "expected_bytes": 154332064,
        "is_zip": True,
    },
    {
        "filename": "noisy_testset_wav.zip",
        "expected_bytes": 170578854,
        "is_zip": True,
    },
    {
        "filename": "logfiles.zip",
        "expected_bytes": 115932,
        "is_zip": True,
    },
    {
        "filename": "trainset_28spk_txt.zip",
        "expected_bytes": 6489312,
        "is_zip": True,
    },
    {
        "filename": "testset_txt.zip",
        "expected_bytes": 440277,
        "is_zip": True,
    },
    {
        "filename": "license_text",
        "expected_bytes": 17416,
        "is_zip": False,
    },
]


def compute_sha256(filepath: Path, chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_storage_requirements(target_dir: Path, required_bytes: int) -> Dict[str, Any]:
    """Verifies that the target path has sufficient free storage space."""
    target_dir.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(str(target_dir))
    has_space = free >= required_bytes
    return {
        "target_path": str(target_dir),
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "required_bytes": required_bytes,
        "has_sufficient_space": has_space,
    }


def download_file_with_progress(
    url: str,
    dest_path: Path,
    expected_bytes: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Downloads a file via streaming HTTP GET with progress updates."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    start_time = time.time()
    downloaded = 0
    last_log_time = start_time

    print(f"Downloading {dest_path.name} from {url}...")
    with urllib.request.urlopen(req, timeout=60) as resp:
        content_length = resp.headers.get("Content-Length")
        total_size = int(content_length) if content_length else expected_bytes

        with open(temp_path, "wb") as out_f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                curr_time = time.time()
                if curr_time - last_log_time >= 5.0 or (total_size and downloaded >= total_size):
                    elapsed = curr_time - start_time
                    speed_mb = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    if total_size:
                        pct = (downloaded / total_size) * 100
                        print(
                            f"  -> {dest_path.name}: {downloaded / (1024**2):.1f}/{total_size / (1024**2):.1f} MB "
                            f"({pct:.1f}%) - {speed_mb:.2f} MB/s"
                        )
                    else:
                        print(f"  -> {dest_path.name}: {downloaded / (1024**2):.1f} MB - {speed_mb:.2f} MB/s")
                    last_log_time = curr_time

    elapsed = time.time() - start_time
    # Rename atomically
    temp_path.replace(dest_path)
    print(f"  Completed {dest_path.name}: {downloaded / (1024**2):.2f} MB in {elapsed:.1f}s")
    return downloaded


def verify_archive_integrity(dest_path: Path, is_zip: bool, expected_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Validates file existence, size, and zip structure."""
    if not dest_path.exists():
        raise FileNotFoundError(f"File not found: {dest_path}")

    actual_size = dest_path.stat().st_size
    if expected_bytes is not None and actual_size != expected_bytes:
        raise ValueError(
            f"Size mismatch for {dest_path.name}: expected {expected_bytes} bytes, got {actual_size} bytes."
        )

    zip_status = "not_applicable"
    corrupt_file = None
    if is_zip:
        with zipfile.ZipFile(dest_path, "r") as zf:
            corrupt_file = zf.testzip()
            if corrupt_file is not None:
                raise ValueError(f"Corrupt entry detected in zip archive {dest_path.name}: {corrupt_file}")
            zip_status = "valid"

    sha256 = compute_sha256(dest_path)
    return {
        "filename": dest_path.name,
        "path": str(dest_path),
        "size_bytes": actual_size,
        "size_mb": round(actual_size / (1024 * 1024), 2),
        "is_zip": is_zip,
        "zip_status": zip_status,
        "sha256": sha256,
    }


def download_edinburgh_dataset(
    target_dir: Path,
    base_url: str = EDINBURGH_BASE_URL,
    force: bool = False,
) -> Dict[str, Any]:
    """Downloads all official Edinburgh bitstreams and verifies archive integrity."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    total_expected_bytes = sum(b["expected_bytes"] for b in OFFICIAL_BITSTREAMS)
    storage_check = check_storage_requirements(target_dir, total_expected_bytes)

    if not storage_check["has_sufficient_space"]:
        free_gb = storage_check["free_bytes"] / (1024 ** 3)
        req_gb = total_expected_bytes / (1024 ** 3)
        raise RuntimeError(
            f"Insufficient disk space in {target_dir}. Required: {req_gb:.2f} GB, Free: {free_gb:.2f} GB"
        )

    print(f"Target directory: {target_dir}")
    print(f"Total dataset download size: {total_expected_bytes / (1024**3):.2f} GB")
    print(f"Free disk space in target: {storage_check['free_bytes'] / (1024**3):.2f} GB")

    results = {}
    for item in OFFICIAL_BITSTREAMS:
        fname = item["filename"]
        expected_bytes = item["expected_bytes"]
        is_zip = item["is_zip"]
        dest = target_dir / fname
        url = base_url + fname

        # Check if already downloaded and valid
        download_needed = force or not dest.exists()
        if dest.exists() and not force:
            if dest.stat().st_size != expected_bytes:
                print(f"File {fname} exists but size mismatch ({dest.stat().st_size} != {expected_bytes}). Re-downloading...")
                download_needed = True
            else:
                print(f"File {fname} already exists with exact expected size ({dest.stat().st_size:,} bytes). Verifying...")

        if download_needed:
            download_file_with_progress(url, dest, expected_bytes=expected_bytes)

        verify_info = verify_archive_integrity(dest, is_zip=is_zip, expected_bytes=expected_bytes)
        results[fname] = verify_info
        print(f"Verified {fname}: SHA256={verify_info['sha256'][:16]}..., status={verify_info['zip_status']}")

    summary = {
        "status": "SUCCESS",
        "target_dir": str(target_dir),
        "total_files": len(results),
        "total_bytes": sum(r["size_bytes"] for r in results.values()),
        "files": results,
    }
    return summary
