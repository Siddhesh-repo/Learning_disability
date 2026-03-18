"""
Phase 2: Handwriting model training from real-image manifest.

This pipeline consumes the Phase 1 manifest, extracts handwriting features from
each labeled image, builds a trainable feature dataset, and trains a
handwriting-only classifier (normal vs dyslexia-risk).

Usage:
    python train_phase2_handwriting.py
    python train_phase2_handwriting.py --manifest data/processed/handwriting_manifest_labeled_phase1.csv
    python train_phase2_handwriting.py --max-samples 400 --model-types random_forest svm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.training_visualizations import (
    save_learning_curves,
    save_roc_curves,
    save_confusion_matrix,
    save_feature_importance,
    save_model_comparison,
)

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from cv.handwriting_analyzer import HandwritingAnalyzer
from ml.feature_engineering import FeatureEngineer
from ml.disability_predictor import DisabilityPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 handwriting training")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/handwriting_manifest_labeled_phase1.csv",
        help="Path to Phase 1 manifest CSV",
    )
    parser.add_argument(
        "--dataset-output",
        type=str,
        default=None,
        help="Optional output path for extracted handwriting feature CSV",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/phase2_handwriting",
        help="Directory for Phase 2 model artifacts",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on total labeled samples used for fast experimentation",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["random_forest", "gradient_boosting", "svm", "neural_network"],
        help="Model types to train",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable grid-search tuning for each selected model",
    )
    return parser.parse_args()


def _ensure_labeled(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df[df["label"].isin(["yes", "no"])].copy()
    if labeled.empty:
        raise ValueError("Manifest has no labeled rows (yes/no).")
    return labeled


def _downsample_stratified(df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df

    yes_df = df[df["label"] == "yes"]
    no_df = df[df["label"] == "no"]

    yes_target = int(round(max_samples * (len(yes_df) / len(df))))
    no_target = max_samples - yes_target

    yes_sample = yes_df.sample(n=min(len(yes_df), yes_target), random_state=42)
    no_sample = no_df.sample(n=min(len(no_df), no_target), random_state=42)

    sampled = pd.concat([yes_sample, no_sample], axis=0)
    return sampled.sample(frac=1.0, random_state=42).reset_index(drop=True)


def build_feature_dataset(manifest_path: Path, max_samples: int | None = None) -> tuple[pd.DataFrame, dict]:
    manifest = pd.read_csv(manifest_path)

    required = {"sample_id", "label", "stored_path", "source"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    manifest = _ensure_labeled(manifest)
    manifest = _downsample_stratified(manifest, max_samples)

    analyzer = HandwritingAnalyzer()
    rows: list[dict] = []
    skipped = 0

    for rec in manifest.to_dict(orient="records"):
        image_path = Path(rec["stored_path"])
        if not image_path.exists():
            skipped += 1
            continue

        try:
            features = analyzer.extract_features(str(image_path)).to_dict()
            quality = analyzer.check_image_quality(str(image_path))
            overall_score = analyzer.calculate_overall_score(analyzer.features)
        except Exception:
            skipped += 1
            continue

        # Map Phase 1 handwriting labels to Phase 2 binary condition labels.
        condition = "dyslexia" if rec["label"] == "yes" else "normal"

        rows.append(
            {
                "sample_id": rec["sample_id"],
                "source": rec.get("source", "unknown"),
                "label": rec["label"],
                "condition": condition,
                "age": 8,
                "handwriting_overall_score": overall_score,
                "image_sharpness": quality.sharpness,
                "image_contrast": quality.contrast,
                **features,
            }
        )

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise RuntimeError("No usable handwriting feature rows extracted from manifest.")

    stats = {
        "input_rows": int(len(manifest)),
        "extracted_rows": int(len(dataset)),
        "skipped_rows": int(skipped),
        "label_counts": {k: int(v) for k, v in dataset["label"].value_counts().to_dict().items()},
        "condition_counts": {k: int(v) for k, v in dataset["condition"].value_counts().to_dict().items()},
        "source_counts": {k: int(v) for k, v in dataset["source"].value_counts().to_dict().items()},
    }
    return dataset, stats


def train_phase2(dataset: pd.DataFrame, models_dir: Path, model_types: list[str], tune: bool, test_size: float) -> dict:
    # Keep bookkeeping columns in saved CSV, but exclude them from model features.
    dataset_for_training = dataset.drop(columns=["source", "label"], errors="ignore")

    engineer = FeatureEngineer()
    X, y, selected = engineer.fit_transform(dataset_for_training)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    labels = list(engineer.label_encoder.classes_)
    results: dict = {}

    for model_type in model_types:
        predictor = DisabilityPredictor(model_type=model_type)
        if tune:
            tuning = predictor.tune(X_train, y_train)
            history = tuning
        else:
            history = predictor.train(X_train, y_train, X_val, y_val)

        metrics = predictor.evaluate(X_test, y_test, label_names=labels)
        model_path = predictor.save(models_dir)

        viz_dir = models_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        y_pred = predictor.model.predict(X_test)
        save_confusion_matrix(y_test, y_pred, labels, model_type, viz_dir)

        imp = predictor.feature_importance(selected)
        save_feature_importance(imp, model_type, viz_dir)

        save_learning_curves(predictor.model, X, y, model_type, viz_dir)
        save_roc_curves(predictor.model, X_test, y_test, labels, model_type, viz_dir)

        results[model_type] = {
            "history": history,
            "metrics": metrics,
            "model_path": model_path,
        }

    save_model_comparison(results, models_dir / "visualizations")

    engineer.save(models_dir)
    best_name, best = max(results.items(), key=lambda kv: kv[1]["metrics"]["accuracy"])

    (models_dir / "best_model.txt").write_text(
        f"Best Model: {best_name}\n"
        f"Model Path: {Path(best['model_path']).name}\n"
        f"Accuracy: {best['metrics']['accuracy']:.4f}\n"
        f"F1-Score: {best['metrics']['f1_score']:.4f}\n"
    )

    return {
        "labels": labels,
        "selected_features": selected,
        "results": results,
        "best_model": best_name,
        "best_accuracy": best["metrics"]["accuracy"],
        "best_f1": best["metrics"]["f1_score"],
    }


def main() -> None:
    args = parse_args()
    Config.init_dirs()

    manifest_path = Path(args.manifest).expanduser().resolve()
    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = (Path(__file__).resolve().parent / models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PHASE 2 — HANDWRITING MODEL TRAINING")
    print("=" * 70)
    print(f"Manifest: {manifest_path}")

    dataset, dataset_stats = build_feature_dataset(manifest_path, max_samples=args.max_samples)

    dataset_output = (
        Path(args.dataset_output).expanduser().resolve()
        if args.dataset_output
        else (Path(__file__).resolve().parent / "data" / "processed" / "phase2_handwriting_features.csv")
    )
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_output, index=False)

    print(f"Extracted feature dataset: {dataset_output} ({len(dataset)} rows)")
    print(f"Label distribution: {dataset_stats['label_counts']}")

    training_summary = train_phase2(
        dataset=dataset,
        models_dir=models_dir,
        model_types=args.model_types,
        tune=args.tune,
        test_size=args.test_size,
    )

    summary = {
        "phase": "phase2_handwriting",
        "created_at": datetime.now().isoformat(),
        "manifest": str(manifest_path),
        "feature_dataset": str(dataset_output),
        "models_dir": str(models_dir),
        "dataset_stats": dataset_stats,
        "training": training_summary,
    }
    summary_path = models_dir / "phase2_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("-" * 70)
    print(f"Best model: {training_summary['best_model']}")
    print(f"Accuracy: {training_summary['best_accuracy']:.4f}")
    print(f"F1-score: {training_summary['best_f1']:.4f}")
    print(f"Summary: {summary_path}")
    print("-" * 70)


if __name__ == "__main__":
    main()
