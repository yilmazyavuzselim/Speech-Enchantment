"""Dataset splitting, verification, and manifest creation for Edinburgh VoiceBank-DEMAND."""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional
import soundfile as sf
import numpy as np


def parse_log_file(log_path: Path) -> List[Dict[str, Any]]:
    """Parses Edinburgh log file (wavfile noise SNR(dB))."""
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            wav_name = parts[0]
            if not wav_name.endswith(".wav"):
                wav_name = wav_name + ".wav"
            noise_type = parts[1]
            try:
                snr_db = float(parts[2])
            except ValueError:
                snr_db = float("nan")

            stem = Path(wav_name).stem
            speaker_id = stem.split("_")[0] if "_" in stem else "unknown"

            entries.append({
                "wav_name": wav_name,
                "utterance_id": stem,
                "speaker_id": speaker_id,
                "noise_type": noise_type,
                "snr_db": snr_db,
            })
    return entries


def validate_single_audio_pair(
    clean_path: Path,
    noisy_path: Path,
    expected_sr: int = 16000,
) -> Tuple[int, float, Optional[str]]:
    """Validates an audio pair for existence, sample rate, channels, finite values, and sample alignment."""
    if not clean_path.exists():
        return 0, 0.0, f"Clean audio missing: {clean_path}"
    if not noisy_path.exists():
        return 0, 0.0, f"Noisy audio missing: {noisy_path}"

    try:
        clean_info = sf.info(str(clean_path))
        noisy_info = sf.info(str(noisy_path))
    except Exception as e:
        return 0, 0.0, f"Error reading audio headers: {e}"

    if clean_info.samplerate != expected_sr:
        return 0, 0.0, f"Clean sample rate {clean_info.samplerate} != {expected_sr}"
    if noisy_info.samplerate != expected_sr:
        return 0, 0.0, f"Noisy sample rate {noisy_info.samplerate} != {expected_sr}"
    if clean_info.channels != 1:
        return 0, 0.0, f"Clean channel count {clean_info.channels} != 1"
    if noisy_info.channels != 1:
        return 0, 0.0, f"Noisy channel count {noisy_info.channels} != 1"
    if clean_info.frames != noisy_info.frames:
        return 0, 0.0, f"Sample count mismatch: clean={clean_info.frames}, noisy={noisy_info.frames}"

    try:
        clean_data, _ = sf.read(str(clean_path), dtype="float32")
        noisy_data, _ = sf.read(str(noisy_path), dtype="float32")
    except Exception as e:
        return 0, 0.0, f"Error reading audio data: {e}"

    if not np.isfinite(clean_data).all():
        return 0, 0.0, "Non-finite (NaN/Inf) values found in clean audio"
    if not np.isfinite(noisy_data).all():
        return 0, 0.0, "Non-finite (NaN/Inf) values found in noisy audio"

    duration = clean_info.frames / expected_sr
    return clean_info.frames, duration, None


def save_manifest_csv(manifest: List[Dict[str, Any]], out_path: Path) -> None:
    """Saves a manifest as CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest:
        return
    fieldnames = list(manifest[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def save_manifest_json(manifest: List[Dict[str, Any]], out_path: Path) -> None:
    """Saves a manifest as JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def create_manifests_and_validate(
    data_16k_dir: Path,
    manifests_dir: Path,
    report_path: Path,
    val_speakers: List[str] = ["p226", "p287"],
    test_speakers: List[str] = ["p232", "p257"],
    expected_sr: int = 16000,
) -> Dict[str, Any]:
    """Generates train, validation, and test manifests and runs rigorous full-dataset validation."""
    data_16k_dir = Path(data_16k_dir)
    manifests_dir = Path(manifests_dir)
    report_path = Path(report_path)

    # Locate log files
    train_log_candidates = [
        data_16k_dir / "logfiles" / "log_trainset_28spk.txt",
        data_16k_dir / "log_trainset_28spk.txt",
        data_16k_dir / "logfiles.zip",
    ]
    test_log_candidates = [
        data_16k_dir / "logfiles" / "log_testset.txt",
        data_16k_dir / "log_testset.txt",
        data_16k_dir / "testset_txt" / "log_testset.txt",
    ]

    train_log_path = next((p for p in train_log_candidates if p.exists() and p.is_file()), None)
    test_log_path = next((p for p in test_log_candidates if p.exists() and p.is_file()), None)

    if not train_log_path or not test_log_path:
        # Check subdirectories
        train_logs = list(data_16k_dir.glob("**/log_trainset_28spk.txt"))
        test_logs = list(data_16k_dir.glob("**/log_testset.txt"))
        if train_logs:
            train_log_path = train_logs[0]
        if test_logs:
            test_log_path = test_logs[0]

    if not train_log_path or not test_log_path:
        raise FileNotFoundError(
            f"Log files not found in {data_16k_dir}. Looked for log_trainset_28spk.txt and log_testset.txt."
        )

    print(f"Parsing train log: {train_log_path}")
    raw_train_entries = parse_log_file(train_log_path)
    print(f"  Parsed {len(raw_train_entries)} training entries.")

    print(f"Parsing test log: {test_log_path}")
    raw_test_entries = parse_log_file(test_log_path)
    print(f"  Parsed {len(raw_test_entries)} test entries.")

    # Speaker inspection
    train_speakers_found = sorted(set(e["speaker_id"] for e in raw_train_entries))
    test_speakers_found = sorted(set(e["speaker_id"] for e in raw_test_entries))

    print(f"\nSpeakers found in 28-speaker training log ({len(train_speakers_found)}): {train_speakers_found}")
    print(f"Speakers found in test log ({len(test_speakers_found)}): {test_speakers_found}")

    # Verify validation speakers exist in training set
    for val_spk in val_speakers:
        if val_spk not in train_speakers_found:
            raise ValueError(f"Configured validation speaker '{val_spk}' not found in training set!")

    # Verify test speakers are NOT in training set
    overlap = set(test_speakers).intersection(set(train_speakers_found))
    if overlap:
        raise ValueError(f"FATAL: Test speakers overlap with training set: {overlap}!")

    clean_train_dir = data_16k_dir / "clean_trainset_28spk_wav"
    noisy_train_dir = data_16k_dir / "noisy_trainset_28spk_wav"
    clean_test_dir = data_16k_dir / "clean_testset_wav"
    noisy_test_dir = data_16k_dir / "noisy_testset_wav"

    train_manifest = []
    val_manifest = []
    test_manifest = []
    anomalies = []

    print("\nValidating and constructing Train and Validation splits...")
    val_spk_set = set(val_speakers)
    for entry in raw_train_entries:
        wav_name = entry["wav_name"]
        spk = entry["speaker_id"]
        c_path = clean_train_dir / wav_name
        n_path = noisy_train_dir / wav_name

        num_samples, duration_sec, error = validate_single_audio_pair(c_path, n_path, expected_sr)
        if error:
            anomalies.append({"split": "train_candidate", "wav_name": wav_name, "error": error})

        row = {
            "utterance_id": entry["utterance_id"],
            "speaker_id": spk,
            "clean_path": str(c_path.resolve()),
            "noisy_path": str(n_path.resolve()),
            "clean_rel_path": str(c_path.relative_to(data_16k_dir)),
            "noisy_rel_path": str(n_path.relative_to(data_16k_dir)),
            "noise_type": entry["noise_type"],
            "snr_db": entry["snr_db"],
            "num_samples": num_samples,
            "duration_sec": round(duration_sec, 4),
            "sample_rate": expected_sr,
        }

        if spk in val_spk_set:
            val_manifest.append(row)
        else:
            train_manifest.append(row)

    print("Validating and constructing Test split...")
    for entry in raw_test_entries:
        wav_name = entry["wav_name"]
        spk = entry["speaker_id"]
        c_path = clean_test_dir / wav_name
        n_path = noisy_test_dir / wav_name

        num_samples, duration_sec, error = validate_single_audio_pair(c_path, n_path, expected_sr)
        if error:
            anomalies.append({"split": "test", "wav_name": wav_name, "error": error})

        row = {
            "utterance_id": entry["utterance_id"],
            "speaker_id": spk,
            "clean_path": str(c_path.resolve()),
            "noisy_path": str(n_path.resolve()),
            "clean_rel_path": str(c_path.relative_to(data_16k_dir)),
            "noisy_rel_path": str(n_path.relative_to(data_16k_dir)),
            "noise_type": entry["noise_type"],
            "snr_db": entry["snr_db"],
            "num_samples": num_samples,
            "duration_sec": round(duration_sec, 4),
            "sample_rate": expected_sr,
        }
        test_manifest.append(row)

    print(f"\nFinal Split Utterance Counts:")
    print(f"  Train : {len(train_manifest)} utterances ({len(set(r['speaker_id'] for r in train_manifest))} speakers)")
    print(f"  Val   : {len(val_manifest)} utterances ({len(set(r['speaker_id'] for r in val_manifest))} speakers: {val_speakers})")
    print(f"  Test  : {len(test_manifest)} utterances ({len(set(r['speaker_id'] for r in test_manifest))} speakers: {test_speakers})")
    print(f"  Total : {len(train_manifest) + len(val_manifest) + len(test_manifest)} utterances")
    print(f"  Total Anomalies: {len(anomalies)}")

    if anomalies:
        print(f"WARNING: Detected {len(anomalies)} anomalies!")
        for a in anomalies[:10]:
            print("  ", a)

    # Save manifest files
    manifests_dir.mkdir(parents=True, exist_ok=True)
    save_manifest_json(train_manifest, manifests_dir / "train_manifest.json")
    save_manifest_csv(train_manifest, manifests_dir / "train_manifest.csv")
    save_manifest_json(val_manifest, manifests_dir / "val_manifest.json")
    save_manifest_csv(val_manifest, manifests_dir / "val_manifest.csv")
    save_manifest_json(test_manifest, manifests_dir / "test_manifest.json")
    save_manifest_csv(test_manifest, manifests_dir / "test_manifest.csv")

    # Metrics aggregation
    def compute_stats(manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
        durations = [r["duration_sec"] for r in manifest]
        total_sec = sum(durations)
        speakers = sorted(set(r["speaker_id"] for r in manifest))
        spk_counts = {spk: sum(1 for r in manifest if r["speaker_id"] == spk) for spk in speakers}
        spk_durations = {spk: round(sum(r["duration_sec"] for r in manifest if r["speaker_id"] == spk) / 60, 2) for spk in speakers}
        noise_types = sorted(set(r["noise_type"] for r in manifest))
        noise_counts = {n: sum(1 for r in manifest if r["noise_type"] == n) for n in noise_types}
        snr_vals = [r["snr_db"] for r in manifest]
        snr_counts = {}
        for snr in sorted(set(snr_vals)):
            snr_counts[str(snr)] = sum(1 for r in manifest if r["snr_db"] == snr)

        return {
            "num_utterances": len(manifest),
            "total_duration_sec": round(total_sec, 2),
            "total_duration_hours": round(total_sec / 3600, 3),
            "num_speakers": len(speakers),
            "speakers": speakers,
            "speaker_utterance_counts": spk_counts,
            "speaker_duration_minutes": spk_durations,
            "noise_types": noise_counts,
            "snr_db_distribution": snr_counts,
            "min_duration_sec": round(min(durations), 3) if durations else 0,
            "max_duration_sec": round(max(durations), 3) if durations else 0,
            "mean_duration_sec": round(np.mean(durations), 3) if durations else 0,
        }

    report = {
        "status": "VALIDATED" if len(anomalies) == 0 else "VALIDATION_FAILED",
        "sample_rate_hz": expected_sr,
        "splits": {
            "train": compute_stats(train_manifest),
            "val": compute_stats(val_manifest),
            "test": compute_stats(test_manifest),
            "all": {
                "total_utterances": len(train_manifest) + len(val_manifest) + len(test_manifest),
                "total_duration_hours": round(
                    (sum(r["duration_sec"] for r in train_manifest) +
                     sum(r["duration_sec"] for r in val_manifest) +
                     sum(r["duration_sec"] for r in test_manifest)) / 3600, 3
                ),
            },
        },
        "speaker_isolation_check": {
            "train_val_speakers": train_speakers_found,
            "test_speakers": test_speakers_found,
            "overlap": list(overlap),
            "is_disjoint": len(overlap) == 0,
        },
        "anomalies_detected": anomalies,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved validation report to {report_path}")

    return report
