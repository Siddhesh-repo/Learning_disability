"""
Multi-model disability predictor.

Supports Random Forest, Gradient Boosting, SVM, and MLP.
Trained models are persisted with metadata for auditability.
"""
import json
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from config import Config

MODEL_REGISTRY = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42,
    ),
    "svm": lambda: SVC(
        kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42,
    ),
    "neural_network": lambda: MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), activation="relu", solver="adam",
        learning_rate="adaptive", max_iter=500, random_state=42,
    ),
}

TUNING_GRIDS = {
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "svm": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto", 0.01],
        "kernel": ["rbf", "linear"],
    },
    "neural_network": {
        "hidden_layer_sizes": [(64, 32), (128, 64, 32), (64, 32, 16)],
        "learning_rate": ["constant", "adaptive"],
        "alpha": [0.0001, 0.001, 0.01],
    },
}


class DisabilityPredictor:
    """Train, evaluate, and persist a disability screening model."""

    def __init__(self, model_type: str = "random_forest"):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. "
                             f"Choose from {list(MODEL_REGISTRY)}")
        self.model_type = model_type
        self.model = MODEL_REGISTRY[model_type]()
        self.is_trained = False
        self.history: Dict = {}

    # ── Training ─────────────────────────────────────────────
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict:
        self.model.fit(X_train, y_train)
        self.is_trained = True

        metrics = {"train_accuracy": accuracy_score(y_train, self.model.predict(X_train))}
        if X_val is not None:
            metrics["val_accuracy"] = accuracy_score(y_val, self.model.predict(X_val))
        self.history = metrics
        return metrics

    def tune(self, X_train, y_train) -> Dict:
        """Hyperparameter tuning via GridSearchCV."""
        grid = GridSearchCV(
            self.model, TUNING_GRIDS.get(self.model_type, {}),
            cv=5, scoring="accuracy", n_jobs=-1, verbose=0,
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.is_trained = True
        return {"best_params": grid.best_params_, "best_score": grid.best_score_}

    # ── Inference ────────────────────────────────────────────
    def predict(self, X) -> np.ndarray:
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        self._check_trained()
        return self.model.predict_proba(X)

    # ── Evaluation ───────────────────────────────────────────
    def evaluate(self, X_test, y_test, label_names: List[str] | None = None) -> Dict:
        self._check_trained()
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        cv = cross_val_score(self.model, X_test, y_test, cv=min(5, len(y_test)))

        if label_names is None:
            label_names = [f"class_{i}" for i in range(len(np.unique(y_test)))]

        report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)

        return {
            "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
        }

    def feature_importance(self, feature_names: list) -> list[dict] | None:
        """Return sorted feature importances (tree models only)."""
        if not hasattr(self.model, "feature_importances_"):
            return None
        imp = self.model.feature_importances_
        pairs = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]

    # ── Persistence ──────────────────────────────────────────
    def save(self, out_dir: Path | str | None = None) -> str:
        out = Path(out_dir or Config.MODELS_DIR)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = out / f"{self.model_type}_{ts}.pkl"
        joblib.dump(self.model, model_file)

        meta = {
            "model_type": self.model_type,
            "created": ts,
            "is_trained": self.is_trained,
            "history": self.history,
        }
        meta_file = out / f"{self.model_type}_{ts}_metadata.json"
        meta_file.write_text(json.dumps(meta, indent=2, default=str))

        return str(model_file)

    def load(self, model_path: str | Path):
        self.model = joblib.load(model_path)
        self.is_trained = True

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
