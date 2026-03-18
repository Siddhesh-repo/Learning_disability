"""
Run implemented project phases in sequence.

Current coverage:
- Phase 2: handwriting model training from Phase 1 manifest
- Phase 3: speech model training (synthetic bootstrap or real CSV)
- Phase 4: fusion model training (synthetic bootstrap or real merged CSVs)

Usage:
    python run_all_phases.py
    python run_all_phases.py --skip-phase3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run implemented phases")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    parser.add_argument("--skip-phase4", action="store_true")
    parser.add_argument("--tune", action="store_true", help="Enable tuning for all phases")
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["random_forest"],
        help="Model types to train in each phase",
    )
    parser.add_argument(
        "--phase2-manifest",
        type=str,
        default="data/processed/handwriting_manifest_labeled_phase1.csv",
        help="Phase 2 input manifest",
    )
    parser.add_argument("--phase2-max-samples", type=int, default=None)
    parser.add_argument("--phase3-dataset", type=str, default=None, help="Real speech feature CSV for Phase 3")
    parser.add_argument("--phase3-synthetic-samples", type=int, default=800)
    parser.add_argument("--phase4-handwriting-csv", type=str, default=None, help="Real handwriting feature CSV for Phase 4")
    parser.add_argument("--phase4-speech-csv", type=str, default=None, help="Real speech feature CSV for Phase 4")
    parser.add_argument("--phase4-synthetic-samples", type=int, default=1000)
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent

    if not args.skip_phase2:
        phase2_cmd = [
            args.python,
            str(here / "train_phase2_handwriting.py"),
            "--manifest",
            args.phase2_manifest,
            "--model-types",
            *args.model_types,
        ]
        if args.phase2_max_samples is not None:
            phase2_cmd.extend(["--max-samples", str(args.phase2_max_samples)])
        if args.tune:
            phase2_cmd.append("--tune")
        run_cmd(phase2_cmd)

    if not args.skip_phase3:
        phase3_cmd = [
            args.python,
            str(here / "train_phase3_speech.py"),
            "--model-types",
            *args.model_types,
        ]
        if args.phase3_dataset:
            phase3_cmd.extend(["--dataset", args.phase3_dataset])
        else:
            phase3_cmd.extend(["--synthetic-samples", str(args.phase3_synthetic_samples)])
        if args.tune:
            phase3_cmd.append("--tune")
        run_cmd(phase3_cmd)

    if not args.skip_phase4:
        phase4_cmd = [
            args.python,
            str(here / "train_phase4_fusion.py"),
            "--model-types",
            *args.model_types,
        ]
        if args.phase4_handwriting_csv and args.phase4_speech_csv:
            phase4_cmd.extend(["--handwriting-csv", args.phase4_handwriting_csv])
            phase4_cmd.extend(["--speech-csv", args.phase4_speech_csv])
        else:
            phase4_cmd.extend(["--synthetic-samples", str(args.phase4_synthetic_samples)])
        if args.tune:
            phase4_cmd.append("--tune")
        run_cmd(phase4_cmd)

    print("\nAll selected phases completed.")


if __name__ == "__main__":
    main()
