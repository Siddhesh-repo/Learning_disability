"""
Training visualization utilities.

Generates learning curves, ROC-AUC curves, confusion matrices,
and model comparison charts. Saves all as PNG files to models/visualizations/.
"""
import base64
import io
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

logger = logging.getLogger(__name__)


def save_learning_curves(
    model, X, y, model_name: str, output_dir: Path, cv: int = 5
):
    """Generate and save learning curves showing train vs validation accuracy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
        ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training Score")
        ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation Score")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning Curve — {model_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        path = output_dir / f"learning_curve_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Learning curve saved: %s", path)
        return str(path)
    except Exception:
        logger.exception("Failed to generate learning curve for %s", model_name)
        return None


def save_roc_curves(
    model, X_test, y_test, label_names: list, model_name: str, output_dir: Path
):
    """Generate and save per-class ROC-AUC curves."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        y_proba = model.predict_proba(X_test)
        n_classes = len(label_names)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for i, (label, color) in enumerate(zip(label_names, colors)):
            if n_classes == 2 and i == 0:
                continue  # Skip negative class for binary
            y_binary = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {model_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        path = output_dir / f"roc_curves_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("ROC curves saved: %s", path)
        return str(path)
    except Exception:
        logger.exception("Failed to generate ROC curves for %s", model_name)
        return None


def save_confusion_matrix(
    y_true, y_pred, label_names: list, model_name: str, output_dir: Path
):
    """Generate and save an enhanced confusion matrix with percentages."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=label_names, yticklabels=label_names)
        axes[0].set_title(f"Confusion Matrix (Counts) — {model_name}")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        # Percentages
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
                    xticklabels=label_names, yticklabels=label_names)
        axes[1].set_title(f"Confusion Matrix (%) — {model_name}")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")

        plt.tight_layout()
        path = output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved: %s", path)
        return str(path)
    except Exception:
        logger.exception("Failed to generate confusion matrix for %s", model_name)
        return None


def save_model_comparison(results: dict, output_dir: Path):
    """
    Generate and save a model comparison bar chart.

    Parameters
    ----------
    results : dict
        model_name -> {"metrics": {"accuracy": ..., "f1_score": ..., ...}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_names = list(results.keys())
        metrics_to_show = ["accuracy", "f1_score", "precision", "recall"]
        existing = [
            m for m in metrics_to_show
            if m in next(iter(results.values()))["metrics"]
        ]

        x = np.arange(len(model_names))
        width = 0.18
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, metric in enumerate(existing):
            values = [results[m]["metrics"][metric] for m in model_names]
            bars = ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title())
            ax.bar_label(bars, fmt="%.3f", fontsize=7)

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.set_xticks(x + width * (len(existing) - 1) / 2)
        ax.set_xticklabels([n.replace("_", " ").title() for n in model_names], rotation=15)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path = output_dir / "model_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Model comparison chart saved: %s", path)
        return str(path)
    except Exception:
        logger.exception("Failed to generate model comparison chart")
        return None


def save_feature_importance(
    importances: list, model_name: str, output_dir: Path, top_n: int = 15
):
    """Generate and save a horizontal bar chart of feature importances."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        top = importances[:top_n]
        names = [e["feature"].replace("_", " ").title() for e in top]
        values = [e["importance"] for e in top]

        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(names))))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {model_name}")
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        path = output_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Feature importance chart saved: %s", path)
        return str(path)
    except Exception:
        logger.exception("Failed to generate feature importance chart for %s", model_name)
        return None
