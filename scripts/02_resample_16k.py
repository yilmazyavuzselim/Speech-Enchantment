"""Script 02: Resample Edinburgh VoiceBank-DEMAND to 16 kHz Mono and Package Tar."""

import argparse
import json
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import resample_dataset_to_16k


def parse_args():
    parser = argparse.ArgumentParser(description="Resample dataset to 16 kHz mono.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--raw-zips-dir",
        type=str,
        default=None,
        help="Override path to 48 kHz raw archives directory.",
    )
    parser.add_argument(
        "--local-extract-dir",
        type=str,
        default="/content/raw_48k_extracted",
        help="Temporary local scratch directory for 48 kHz extraction.",
    )
    parser.add_argument(
        "--local-16k-dir",
        type=str,
        default=None,
        help="Override destination directory for 16 kHz resampled dataset.",
    )
    parser.add_argument(
        "--tar-output",
        type=str,
        default=None,
        help="Override output path for persistent edinburgh_16k.tar archive.",
    )
    parser.add_argument(
        "--no-tar",
        action="store_true",
        help="Skip creating the tar archive.",
    )
    parser.add_argument(
        "--keep-temp-48k",
        action="store_true",
        help="Do not delete temporary 48 kHz extracted directory after resampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    raw_zips_dir = Path(args.raw_zips_dir or paths_cfg.get("raw_zips_dir", "/content/drive/MyDrive/Speech-Enhancement/data/raw_48k_zips"))
    local_extract_dir = Path(args.local_extract_dir)
    local_16k_dir = Path(args.local_16k_dir or paths_cfg.get("local_data_16k_dir", "/content/data_16k"))
    tar_output = None if args.no_tar else Path(args.tar_output or paths_cfg.get("tar_16k_path", "/content/drive/MyDrive/Speech-Enhancement/data/edinburgh_16k.tar"))

    target_sr = cfg.get("dataset", {}).get("sample_rate", 16000)
    orig_sr = cfg.get("dataset", {}).get("orig_sample_rate", 48000)

    print("=" * 72)
    print("SCRIPT 02: RESAMPLE DATASET TO 16 kHz MONO & TAR PACKAGING")
    print("=" * 72)
    print(f"Raw 48 kHz archives dir : {raw_zips_dir}")
    print(f"Temporary extraction dir: {local_extract_dir}")
    print(f"Resampled 16 kHz dir    : {local_16k_dir}")
    print(f"Persistent tar output   : {tar_output}")
    print(f"Sample rate conversion  : {orig_sr} Hz -> {target_sr} Hz (16-bit PCM)")

    report = resample_dataset_to_16k(
        raw_zips_dir=raw_zips_dir,
        local_extract_dir=local_extract_dir,
        local_16k_dir=local_16k_dir,
        output_tar=tar_output,
        orig_sr=orig_sr,
        target_sr=target_sr,
        clean_temp_extracted=not args.keep_temp_48k,
    )

    out_report = PROJECT_ROOT / "results" / "resample_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nResampling pipeline completed successfully.")
    print(f"Saved resample report to {out_report}")


if __name__ == "__main__":
    main()
