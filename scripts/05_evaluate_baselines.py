"""Script 05: Evaluate Speech Enhancement Baselines (Noisy, Spectral Gating, Demucs dns48)."""

import argparse
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import BaselineEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate speech enhancement baselines on val and test sets.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="manifests/val_manifest.json",
        help="Path to validation manifest JSON.",
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        default="manifests/test_manifest.json",
        help="Path to test manifest JSON.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override path to 16 kHz resampled dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation reports and CSVs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "val", "test"],
        default="all",
        help="Which split to evaluate: 'val', 'test', or 'all' (default).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device for Demucs ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for CPU metric computation.",
    )
    return parser.parse_args()


def print_summary_table(summary: dict):
    split_name = summary.get("split", "unknown").upper()
    num_utts = summary.get("num_utterances", 0)
    print("\n" + "=" * 78)
    print(f"BASELINE EVALUATION SUMMARY: {split_name} SET ({num_utts} UTTERANCES)")
    print("=" * 78)
    print(f"{'Condition':<20} | {'PESQ (Mean±Std)':<18} | {'PESQ Median':<12} | {'STOI (Mean±Std)':<18} | {'STOI Median':<12}")
    print("-" * 78)

    for cond, stats in summary["conditions"].items():
        p = stats["pesq_wb"]
        s = stats["stoi"]
        p_str = f"{p['mean']:.4f} ± {p['std']:.4f}" if p["mean"] is not None else "N/A"
        p_med = f"{p['median']:.4f}" if p["median"] is not None else "N/A"
        s_str = f"{s['mean']:.4f} ± {s['std']:.4f}" if s["mean"] is not None else "N/A"
        s_med = f"{s['median']:.4f}" if s["median"] is not None else "N/A"
        print(f"{cond:<20} | {p_str:<18} | {p_med:<12} | {s_str:<18} | {s_med:<12}")
    print("=" * 78)


def main():
    args = parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or cfg.get("paths", {}).get("local_data_16k_dir", "/content/data_16k")
    output_dir = PROJECT_ROOT / Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_manifest = PROJECT_ROOT / Path(args.val_manifest)
    test_manifest = PROJECT_ROOT / Path(args.test_manifest)

    print("=" * 78)
    print("SCRIPT 05: SPEECH ENHANCEMENT BASELINE EVALUATION")
    print("=" * 78)
    print(f"Data Directory  : {data_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Split(s) : {args.split}")
    print(f"Inference Device: {args.device}")

    evaluator = BaselineEvaluator(
        config=cfg,
        device=args.device,
        max_workers=args.workers,
    )

    if args.split in ("val", "all"):
        if not val_manifest.exists():
            raise FileNotFoundError(f"Validation manifest not found: {val_manifest}")
        val_rows, val_summary = evaluator.evaluate_split(
            manifest_path=val_manifest,
            local_data_dir=data_dir,
            split_name="val",
        )
        evaluator.save_results(val_rows, val_summary, output_dir / "val_baseline")
        print_summary_table(val_summary)

    if args.split in ("test", "all"):
        if not test_manifest.exists():
            raise FileNotFoundError(f"Test manifest not found: {test_manifest}")
        test_rows, test_summary = evaluator.evaluate_split(
            manifest_path=test_manifest,
            local_data_dir=data_dir,
            split_name="test",
        )
        evaluator.save_results(test_rows, test_summary, output_dir / "test_baseline")
        print_summary_table(test_summary)

    print("\nAll baseline evaluations completed successfully.")


if __name__ == "__main__":
    main()
