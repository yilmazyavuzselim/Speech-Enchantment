"""Script 06: Fine-Tune Demucs dns48 on Edinburgh VoiceBank-DEMAND."""

import argparse
from pathlib import Path
import sys
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import DemucsTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Demucs dns48 on VoiceBank-DEMAND.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default="manifests/train_manifest.json",
        help="Path to training manifest JSON.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="manifests/val_manifest.json",
        help="Path to validation manifest JSON.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=None,
        help="Override persistent checkpoints directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Override results directory.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run preflight verification and GPU profiling only (no full training).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint.th if available.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs (default 15).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default 16).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default 5e-5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of CPU worker processes for validation metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides if specified
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg.setdefault("training", {})["lr"] = args.lr

    train_manifest = PROJECT_ROOT / Path(args.train_manifest)
    val_manifest = PROJECT_ROOT / Path(args.val_manifest)
    results_dir = PROJECT_ROOT / Path(args.results_dir)

    trainer = DemucsTrainer(
        config=cfg,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        device=args.device,
        checkpoints_dir=args.checkpoints_dir,
        results_dir=results_dir,
        max_workers=args.workers,
    )

    if args.preflight_only:
        print("\nExecuting Stage 1 Controlled Preflight & GPU Profiling...")
        report = trainer.run_preflight()
        print(f"\nPreflight completed successfully. Status: {report['status']}")
        return

    if args.resume:
        trainer.resume_from_checkpoint()

    # Execute full training
    summary = trainer.train()
    print(f"\nFine-tuning completed. Summary saved to {results_dir}")


if __name__ == "__main__":
    main()
