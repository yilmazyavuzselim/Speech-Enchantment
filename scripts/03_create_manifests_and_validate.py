"""Script 03: Create Manifests and Perform Full Dataset Validation."""

import argparse
import json
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import create_manifests_and_validate


def parse_args():
    parser = argparse.ArgumentParser(description="Create manifests and validate resampled dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to 16 kHz resampled dataset root directory.",
    )
    parser.add_argument(
        "--manifests-dir",
        type=str,
        default=None,
        help="Destination directory for manifest files.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Destination path for validation report JSON.",
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
    dataset_cfg = cfg.get("dataset", {})

    data_dir = Path(args.data_dir or paths_cfg.get("local_data_16k_dir", "/content/data_16k"))
    manifests_dir = PROJECT_ROOT / Path(args.manifests_dir or paths_cfg.get("manifests_dir", "manifests"))
    report_path = PROJECT_ROOT / Path(args.report_path or "results/dataset_validation_report.json")

    val_speakers = dataset_cfg.get("val_speakers", ["p226", "p287"])
    test_speakers = dataset_cfg.get("test_speakers", ["p232", "p257"])
    target_sr = dataset_cfg.get("sample_rate", 16000)

    print("=" * 72)
    print("SCRIPT 03: MANIFEST GENERATION AND DATASET VALIDATION")
    print("=" * 72)
    print(f"Dataset 16k Directory : {data_dir}")
    print(f"Manifests Directory   : {manifests_dir}")
    print(f"Validation Report     : {report_path}")
    print(f"Validation Speakers   : {val_speakers}")
    print(f"Test Speakers         : {test_speakers}")
    print(f"Expected Sample Rate  : {target_sr} Hz")

    report = create_manifests_and_validate(
        data_16k_dir=data_dir,
        manifests_dir=manifests_dir,
        report_path=report_path,
        val_speakers=val_speakers,
        test_speakers=test_speakers,
        expected_sr=target_sr,
    )

    print("\n" + "=" * 72)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 72)
    print(f"Status           : {report['status']}")
    print(f"Speaker Isolation: {'PASSED (Zero Overlap)' if report['speaker_isolation_check']['is_disjoint'] else 'FAILED'}")
    print(f"Total Utterances : {report['splits']['all']['total_utterances']}")
    print(f"Total Duration   : {report['splits']['all']['total_duration_hours']} hours")
    print(f"Train Split      : {report['splits']['train']['num_utterances']} utts ({report['splits']['train']['total_duration_hours']} hrs, {report['splits']['train']['num_speakers']} speakers)")
    print(f"Val Split        : {report['splits']['val']['num_utterances']} utts ({report['splits']['val']['total_duration_hours']} hrs, {report['splits']['val']['num_speakers']} speakers)")
    print(f"Test Split       : {report['splits']['test']['num_utterances']} utts ({report['splits']['test']['total_duration_hours']} hrs, {report['splits']['test']['num_speakers']} speakers)")
    print(f"Anomalies        : {len(report['anomalies_detected'])}")
    print("=" * 72)

    if report["status"] != "VALIDATED":
        print(f"ERROR: Dataset validation failed with {len(report['anomalies_detected'])} anomalies!")
        sys.exit(1)


if __name__ == "__main__":
    main()
