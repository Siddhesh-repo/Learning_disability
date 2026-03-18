"""
Phase 4: Multimodal fusion model training.

This script starts Phase 4 by training a fusion model from prepared
handwriting and speech feature datasets.

Input options:
1) Provide both feature CSVs with shared sample_id.
2) If omitted, bootstrap from synthetic multimodal data.

Usage:
    python train_phase4_fusion.py
    python train_phase4_fusion.py --handwriting-csv data/processed/phase2_handwriting_features.csv --speech-csv data/processed/phase3_speech_features.csv
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


HW_FEATURES = [
    "avg_letter_size",
    "line_straightness",
    "letter_spacing",
    "word_spacing",
    "writing_pressure",
    "letter_formation_quality",
    "slant_angle",
    "consistency_score",
    "contour_count",
    "aspect_ratio",
    "age",
]

SP_FEATURES = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 fusion model training")
    parser.add_argument("--handwriting-csv", type=str, default=None, help="Path to handwriting feature CSV")
    parser.add_argument("--speech-csv", type=str, default=None, help="Path to speech feature CSV")
    parser.add_argument("--synthetic-samples", type=int, default=1000, help="Synthetic samples per condition fallback")
    parser.add_argument("--models-dir", type=str, default="models/phase4_fusion")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-types", nargs="+", default=["random_forest", "gradient_boosting", "svm", "neural_network"])
    parser.add_argument("--tune", action="store_true")
    return parser.parse_args()


def _prepare_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.handwriting_csv and args.speech_csv:
        hw = pd.read_csv(args.handwriting_csv)
        sp = pd.read_csv(args.speech_csv)

        if "sample_id" not in hw.columns or "sample_id" not in sp.columns:
            raise ValueError("Both handwriting and speech CSVs must contain sample_id.")
        if "condition" not in hw.columns or "condition" not in sp.columns:
            raise ValueError("Both handwriting and speech CSVs must contain condition.")

        hw_keep = ["sample_id", "condition", *[c for c in HW_FEATURES if c in hw.columns]]
        sp_keep = ["sample_id", "condition", *[c for c in SP_FEATURES if c in sp.columns]]

        hw = hw[hw_keep].copy()
        sp = sp[sp_keep].copy()

        merged = hw.merge(sp, on=["sample_id", "condition"], how="inner")
        source = "real-merged"
    else:
        gen = SyntheticDataGenerator(seed=42)
        merged = gen.generate(n_per_condition=args.synthetic_samples)
        source = "synthetic"

    merged = merged[merged["condition"].isin(["normal", "dyslexia", "dysgraphia"])].reset_index(drop=True)
    if merged.empty:
        raise ValueError("No valid fusion rows available for training.")

    return merged, source


def train_phase4(dataset: pd.DataFrame, models_dir: Path, model_types: list[str], tune: bool, test_size: float) -> dict:
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

    feature_csv = Path(__file__).resolve().parent / "data" / "processed" / "phase4_fusion_features.csv"
    feature_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(feature_csv, index=False)

    training = train_phase4(
        dataset=dataset,
        models_dir=models_dir,
        model_types=args.model_types,
        tune=args.tune,
        test_size=args.test_size,
    )

    summary = {
        "phase": "phase4_fusion",
        "created_at": datetime.now().isoformat(),
        "data_source": source,
        "rows": len(dataset),
        "feature_dataset": str(feature_csv),
        "models_dir": str(models_dir),
        "training": training,
    }
    summary_path = models_dir / "phase4_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("=" * 70)
    print("  PHASE 4 — FUSION MODEL TRAINING")
    print("=" * 70)
    print(f"Data source: {source}")
    print(f"Rows: {len(dataset)}")
    print(f"Best model: {training['best_model']}")
    print(f"Accuracy: {training['best_accuracy']:.4f}")
    print(f"F1-score: {training['best_f1']:.4f}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
