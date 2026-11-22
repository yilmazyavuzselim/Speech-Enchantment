"""Script 01: Download Edinburgh VoiceBank-DEMAND 48 kHz Archives and Verify Integrity."""

import argparse
import json
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import download_edinburgh_dataset, check_storage_requirements, OFFICIAL_BITSTREAMS


def parse_args():
    parser = argparse.ArgumentParser(description="Download Edinburgh VoiceBank-DEMAND dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Override destination directory for 48 kHz raw archives.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check storage and remote bitstreams without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist.",
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

    target_dir_str = args.target_dir or cfg.get("paths", {}).get(
        "raw_zips_dir", "/content/drive/MyDrive/Speech-Enhancement/data/raw_48k_zips"
    )
    target_dir = Path(target_dir_str)

    total_expected = sum(b["expected_bytes"] for b in OFFICIAL_BITSTREAMS)
    print("=" * 72)
    print("SCRIPT 01: EDINBURGH VOICEBANK-DEMAND 48 kHz DATASET DOWNLOAD")
    print("=" * 72)
    print(f"Target Directory: {target_dir}")
    print(f"Total Archives: {len(OFFICIAL_BITSTREAMS)} files ({total_expected / (1024**3):.2f} GB)")

    storage = check_storage_requirements(target_dir, total_expected)
    print(f"Storage Free Space: {storage['free_bytes'] / (1024**3):.2f} GB")
    print(f"Storage Required  : {total_expected / (1024**3):.2f} GB")
    print(f"Sufficient Space  : {'YES' if storage['has_sufficient_space'] else 'NO'}")

    if not storage["has_sufficient_space"]:
        print("ERROR: Insufficient storage space on persistent Google Drive!")
        sys.exit(1)

    if args.check_only:
        print("\n--check-only specified. Storage verified. Exiting without download.")
        return

    report = download_edinburgh_dataset(target_dir=target_dir, force=args.force)

    # Save verification report
    out_report = PROJECT_ROOT / "results" / "download_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nDownload and integrity verification completed successfully.")
    print(f"Saved download report to {out_report}")


if __name__ == "__main__":
    main()
