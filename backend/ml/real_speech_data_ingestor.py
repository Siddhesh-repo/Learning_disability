"""
Build a real speech-feature CSV from labeled audio files.

This utility scans an audio directory, infers condition labels from file/folder
names, extracts speech features via SpeechAnalyzer, and writes a Phase-3-ready
CSV for training.

Usage:
    python ml/real_speech_data_ingestor.py \
      --audio-dir ../disability_ai_engine/samples/speech \
      --output-csv data/processed/phase3_speech_features_real.csv

    python ml/real_speech_data_ingestor.py \
      --audio-dir ../disability_ai_engine/samples/speech \
      --label-map "sample_good=normal,sample_poor=dyslexia" \
      --replicate 20 \
      --output-csv data/processed/phase3_speech_features_real.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nlp.speech_analyzer import SpeechAnalyzer


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
VALID_CONDITIONS = {"normal", "dyslexia", "dysgraphia"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create real speech feature CSV")
    parser.add_argument("--audio-dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/processed/phase3_speech_features_real.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--age",
        type=int,
        default=8,
        help="Default age value to attach to each row",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default="",
        help="Comma-separated substring mappings, e.g. 'good=normal,poor=dyslexia'",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="Duplicate each extracted row N times with suffixed sample_id",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include rows with unresolved labels as condition=unknown (not phase3-trainable)",
    )
    return parser.parse_args()


def _parse_label_map(raw: str) -> list[tuple[str, str]]:
    if not raw.strip():
        return []

    pairs: list[tuple[str, str]] = []
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid label-map item '{part}'. Expected pattern=condition.")
        key, value = [x.strip().lower() for x in part.split("=", 1)]
        if value not in VALID_CONDITIONS:
            raise ValueError(
                f"Invalid condition '{value}' in label-map. Use one of {sorted(VALID_CONDITIONS)}."
            )
        if not key:
            raise ValueError("Label-map pattern cannot be empty.")
        pairs.append((key, value))
    return pairs


def _infer_condition(path: Path, custom_map: list[tuple[str, str]]) -> str:
    hay = str(path).lower()

    for pattern, value in custom_map:
        if pattern in hay:
            return value

    if any(tok in hay for tok in ("dysgraphia", "graphia")):
        return "dysgraphia"
    if any(tok in hay for tok in ("dyslexia", "high potential", "high_potential", "poor", "risk", "yes")):
        return "dyslexia"
    if any(tok in hay for tok in ("normal", "control", "low potential", "low_potential", "good", "no")):
        return "normal"

    return "unknown"


def _iter_audio_files(audio_dir: Path) -> list[Path]:
    files = [p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


def _extract_rows(
    files: list[Path],
    analyzer: SpeechAnalyzer,
    age: int,
    custom_map: list[tuple[str, str]],
    include_unknown: bool,
    replicate: int,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    skipped_analysis = 0
    skipped_unknown = 0

    for idx, path in enumerate(files, start=1):
        condition = _infer_condition(path, custom_map)
        if condition == "unknown" and not include_unknown:
            skipped_unknown += 1
            continue

        try:
            f = analyzer.analyze(str(path))
        except Exception:
            skipped_analysis += 1
            continue

        base = {
            "sample_id": f"speech_{idx:05d}",
            "condition": condition,
            "source_path": str(path),
            "age": int(age),
            "reading_speed_wpm": float(f.reading_speed_wpm),
            "pause_frequency": float(f.pause_frequency),
            "average_pause_duration": float(f.average_pause_duration),
            "pronunciation_score": float(f.pronunciation_score),
            "fluency_score": float(f.fluency_score),
            "volume_consistency": float(f.volume_consistency),
            "pitch_variation": float(f.pitch_variation),
            "speech_clarity": float(f.speech_clarity),
            "word_count": int(f.word_count),
            "total_duration": float(f.total_duration),
            "transcript": f.transcript,
        }

        reps = max(1, int(replicate))
        for rep in range(reps):
            row = dict(base)
            if reps > 1:
                row["sample_id"] = f"{base['sample_id']}_r{rep + 1:02d}"
            rows.append(row)

    out = pd.DataFrame(rows)

    stats = {
        "input_audio_files": int(len(files)),
        "rows_written": int(len(out)),
        "skipped_unknown": int(skipped_unknown),
        "skipped_analysis": int(skipped_analysis),
        "condition_counts": (
            {k: int(v) for k, v in out["condition"].value_counts().to_dict().items()}
            if not out.empty
            else {}
        ),
    }
    return out, stats


def main() -> None:
    args = parse_args()

    audio_dir = Path(args.audio_dir).expanduser().resolve()
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")

    output_csv = Path(args.output_csv).expanduser()
    if not output_csv.is_absolute():
        output_csv = (Path(__file__).resolve().parents[1] / output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    custom_map = _parse_label_map(args.label_map)
    files = _iter_audio_files(audio_dir)
    if not files:
        raise RuntimeError(f"No audio files found under: {audio_dir}")

    analyzer = SpeechAnalyzer()
    df, stats = _extract_rows(
        files=files,
        analyzer=analyzer,
        age=args.age,
        custom_map=custom_map,
        include_unknown=args.include_unknown,
        replicate=args.replicate,
    )

    if df.empty:
        raise RuntimeError("No rows generated. Check labels or use --include-unknown.")

    # Phase 3 expects labeled rows for these conditions.
    trainable = df[df["condition"].isin(VALID_CONDITIONS)].copy()
    if trainable.empty:
        raise RuntimeError("No trainable rows with conditions in {normal,dyslexia,dysgraphia}.")

    trainable.to_csv(output_csv, index=False)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "audio_dir": str(audio_dir),
        "output_csv": str(output_csv),
        "label_map": custom_map,
        "include_unknown": bool(args.include_unknown),
        "replicate": int(max(1, args.replicate)),
        "stats": stats,
        "trainable_rows": int(len(trainable)),
        "trainable_condition_counts": {k: int(v) for k, v in trainable["condition"].value_counts().to_dict().items()},
    }
    meta_path = output_csv.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=" * 70)
    print("REAL SPEECH CSV CREATED")
    print("=" * 70)
    print(f"Audio dir: {audio_dir}")
    print(f"Output CSV: {output_csv}")
    print(f"Rows: {len(trainable)}")
    print(f"Conditions: {metadata['trainable_condition_counts']}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
