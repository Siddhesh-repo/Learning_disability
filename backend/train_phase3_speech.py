"""
Phase 3: Speech model training pipeline.

This script starts Phase 3 by training a speech-only classifier from either:
1) a real speech-feature CSV (recommended), or
2) a synthetic bootstrap dataset from SyntheticDataGenerator.

Usage:
    python train_phase3_speech.py --synthetic-samples 800
    python train_phase3_speech.py --dataset data/processed/speech_features.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

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
from ml.data_generator import SyntheticDataGenerator
from ml.feature_engineering import FeatureEngineer
from ml.disability_predictor import DisabilityPredictor


SPEECH_FEATURES = [
    "reading_speed_wpm",
    "pause_frequency",
    "average_pause_duration",
    "pronunciation_score",
    "fluency_score",
    "volume_consistency",
    "pitch_variation",
    "speech_clarity",
    "word_count",
    "total_duration",
    "age",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 speech model training")
    parser.add_argument("--dataset", type=str, default=None, help="Path to real speech feature CSV")
    parser.add_argument("--synthetic-samples", type=int, default=800, help="Synthetic samples per condition if no dataset")
    parser.add_argument("--models-dir", type=str, default="models/phase3_speech", help="Directory for artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-types", nargs="+", default=["random_forest", "gradient_boosting", "svm", "neural_network"])
    parser.add_argument("--tune", action="store_true")
    return parser.parse_args()


def _prepare_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.dataset:
        df = pd.read_csv(args.dataset)
        source = "real"
    else:
        gen = SyntheticDataGenerator(seed=42)
        df = gen.generate(n_per_condition=args.synthetic_samples)
        source = "synthetic"

    required = {"condition", *SPEECH_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Speech dataset missing columns: {sorted(missing)}")

    keep_cols = ["sample_id", "condition", *SPEECH_FEATURES]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    out = out[out["condition"].isin(["normal", "dyslexia", "dysgraphia"])].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rows with condition in {normal,dyslexia,dysgraphia}.")

    return out, source


def train_phase3(dataset: pd.DataFrame, models_dir: Path, model_types: list[str], tune: bool, test_size: float) -> dict:
    engineer = FeatureEngineer()
    X, y, selected = engineer.fit_transform(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    labels = list(engineer.label_encoder.classes_)
    results = {}

    for model_type in model_types:
        predictor = DisabilityPredictor(model_type=model_type)
        history = predictor.tune(X_train, y_train) if tune else predictor.train(X_train, y_train, X_val, y_val)
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

        results[model_type] = {"history": history, "metrics": metrics, "model_path": model_path}

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
        "best_model": best_name,
        "best_accuracy": best["metrics"]["accuracy"],
        "best_f1": best["metrics"]["f1_score"],
        "results": results,
    }


def main() -> None:
    args = parse_args()
    Config.init_dirs()

    dataset, source = _prepare_dataset(args)

    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = (Path(__file__).resolve().parent / models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_csv = Path(__file__).resolve().parent / "data" / "processed" / "phase3_speech_features.csv"
    feature_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(feature_csv, index=False)

    summary_training = train_phase3(
        dataset=dataset,
        models_dir=models_dir,
        model_types=args.model_types,
        tune=args.tune,
        test_size=args.test_size,
    )

    summary = {
        "phase": "phase3_speech",
        "created_at": datetime.now().isoformat(),
        "data_source": source,
        "rows": len(dataset),
        "feature_dataset": str(feature_csv),
        "models_dir": str(models_dir),
        "training": summary_training,
    }
    summary_path = models_dir / "phase3_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("=" * 70)
    print("  PHASE 3 — SPEECH MODEL TRAINING")
    print("=" * 70)
    print(f"Data source: {source}")
    print(f"Rows: {len(dataset)}")
    print(f"Best model: {summary_training['best_model']}")
    print(f"Accuracy: {summary_training['best_accuracy']:.4f}")
    print(f"F1-score: {summary_training['best_f1']:.4f}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
