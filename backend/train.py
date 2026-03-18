"""
Model training pipeline.

Generates data (or loads existing), engineers features, trains and
compares multiple ML models, saves the best one, and produces
evaluation visualisations.

Usage:
    python train.py                         # default 1000 samples/class
    python train.py --samples 500 --tune    # with hyper-parameter tuning
"""
import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

from ml.training_visualizations import (
    save_learning_curves,
    save_roc_curves,
    save_confusion_matrix,
    save_model_comparison,
    save_feature_importance,
)

# Ensure this directory is on the path
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from ml.data_generator import SyntheticDataGenerator
from ml.feature_engineering import FeatureEngineer
from ml.disability_predictor import DisabilityPredictor

Config.init_dirs()

VIZ_DIR = Config.MODELS_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


# ── Visualisation helpers loaded from ml.training_visualizations ─


# ── Main pipeline ────────────────────────────────────────
def train_all(n_samples: int = 1000, test_size: float = 0.2,
              perform_tuning: bool = False, dataset_path: str | None = None):

    print("=" * 70)
    print("  LEARNING DISABILITY SCREENING — MODEL TRAINING")
    print("=" * 70)

    # 1. Data
    if dataset_path and Path(dataset_path).exists():
        dataset = pd.read_csv(dataset_path)
        print(f"Loaded dataset: {dataset_path}  ({len(dataset)} rows)")
    else:
        gen = SyntheticDataGenerator(seed=42)
        dataset = gen.generate(n_per_condition=n_samples)
        gen.save(dataset)
        print(f"Generated synthetic dataset: {len(dataset)} rows")

    print(f"Class distribution:\n{dataset['condition'].value_counts().to_string()}\n")

    # 2. Feature engineering
    engineer = FeatureEngineer()
    X, y, selected = engineer.fit_transform(dataset)
    engineer.save()
    print(f"Engineered features: {X.shape[1]}  (selected from {len(engineer.feature_columns)})")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    labels = list(engineer.label_encoder.classes_)
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"Labels: {labels}\n")

    # 4. Train models
    model_types = ["random_forest", "gradient_boosting", "svm", "neural_network"]
    results = {}

    for mt in model_types:
        print("-" * 70)
        print(f"  Training: {mt}")
        print("-" * 70)
        try:
            pred = DisabilityPredictor(model_type=mt)
            if perform_tuning:
                pred.tune(X_train, y_train)
            else:
                pred.train(X_train, y_train, X_val, y_val)

            metrics = pred.evaluate(X_test, y_test, label_names=labels)
            model_path = pred.save()

            y_pred = pred.model.predict(X_test)
            save_confusion_matrix(y_test, y_pred, labels, mt, VIZ_DIR)

            imp = pred.feature_importance(selected)
            save_feature_importance(imp, mt, VIZ_DIR)

            # New visualisations
            save_learning_curves(pred.model, X, y, mt, VIZ_DIR)
            save_roc_curves(pred.model, X_test, y_test, labels, mt, VIZ_DIR)

            results[mt] = {"predictor": pred, "metrics": metrics,
                           "model_path": model_path}
            print(f"  Accuracy: {metrics['accuracy']:.4f}  "
                  f"F1: {metrics['f1_score']:.4f}\n")

        except Exception as exc:
            print(f"  ERROR training {mt}: {exc}\n")

    if not results:
        print("No models trained successfully.")
        return {}

    # 5. Compare and save best
    save_model_comparison(results, VIZ_DIR)
    best_name, best = max(results.items(), key=lambda kv: kv[1]["metrics"]["accuracy"])

    best_path = Config.MODELS_DIR / "best_model.txt"
    best_path.write_text(
        f"Best Model: {best_name}\n"
        f"Model Path: {Path(best['model_path']).name}\n"
        f"Accuracy: {best['metrics']['accuracy']:.4f}\n"
        f"F1-Score: {best['metrics']['f1_score']:.4f}\n"
    )

    print("=" * 70)
    print(f"  Best model: {best_name}  "
          f"(accuracy={best['metrics']['accuracy']:.4f})")
    print(f"  Saved to : {best['model_path']}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train screening models")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per condition")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tune", action="store_true", help="Enable hyper-parameter tuning")
    parser.add_argument("--dataset", type=str, default=None, help="Path to existing CSV")
    args = parser.parse_args()

    train_all(n_samples=args.samples, test_size=args.test_size,
              perform_tuning=args.tune, dataset_path=args.dataset)
