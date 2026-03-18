"""
Feature engineering pipeline.

Creates derived features, scales inputs, selects the best features,
and encodes labels for training and inference.
"""
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

from config import Config


class FeatureEngineer:
    """Transform raw features into model-ready inputs."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selector: SelectKBest | None = None
        self.feature_columns: list[str] | None = None
        self.selected_features: list[str] | None = None

    # ── Derived features ─────────────────────────────────────
    @staticmethod
    def add_derived(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        # Handwriting derived
        if {"avg_letter_size", "contour_count"}.issubset(d.columns):
            d["size_per_contour"] = d["avg_letter_size"] / (d["contour_count"] + 1)
        if {"letter_spacing", "word_spacing"}.issubset(d.columns):
            d["spacing_ratio"] = d["word_spacing"] / (d["letter_spacing"] + 1)
        if {"letter_formation_quality", "consistency_score"}.issubset(d.columns):
            d["quality_consistency"] = d["letter_formation_quality"] * d["consistency_score"] / 100

        # Speech derived
        if {"reading_speed_wpm", "pause_frequency"}.issubset(d.columns):
            d["reading_efficiency"] = d["reading_speed_wpm"] / (d["pause_frequency"] + 0.1)
        if {"pronunciation_score", "fluency_score"}.issubset(d.columns):
            d["speech_quality"] = (d["pronunciation_score"] + d["fluency_score"]) / 2
        if {"word_count", "total_duration"}.issubset(d.columns):
            d["words_per_second"] = d["word_count"] / (d["total_duration"] + 1)

        # Cross-modal
        if {"letter_formation_quality", "pronunciation_score"}.issubset(d.columns):
            d["motor_language_link"] = d["letter_formation_quality"] * 0.5 + d["pronunciation_score"] * 0.5
        if {"consistency_score", "fluency_score"}.issubset(d.columns):
            d["overall_consistency"] = d["consistency_score"] * 0.5 + d["fluency_score"] * 0.5

        # Age-adjusted
        if "age" in d.columns:
            age_norm = (d["age"] - 6) / 6
            if "writing_pressure" in d.columns:
                d["age_adj_pressure"] = d["writing_pressure"] * (1 + age_norm * 0.3)
            if "reading_speed_wpm" in d.columns:
                d["age_adj_reading"] = d["reading_speed_wpm"] / (1 + age_norm * 0.5)

        return d

    # ── Training ─────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame):
        """Fit on training data. Returns (X_selected, y, selected_feature_names)."""
        df = self.add_derived(df)

        exclude = {"sample_id", "condition", "disability_category"}
        self.feature_columns = [c for c in df.columns if c not in exclude]

        X = df[self.feature_columns].fillna(df[self.feature_columns].mean())
        X_scaled = self.scaler.fit_transform(X.values)

        y = self.label_encoder.fit_transform(df["condition"])

        k = min(20, X_scaled.shape[1])
        self.selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.selector.fit_transform(X_scaled, y)

        mask = self.selector.get_support(indices=True)
        self.selected_features = [self.feature_columns[i] for i in mask]

        return X_selected, y, self.selected_features

    # ── Inference ────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a single sample or batch for prediction."""
        if self.feature_columns is None:
            raise RuntimeError("FeatureEngineer has not been fitted yet.")

        df = self.add_derived(df)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X.values)
        return self.selector.transform(X_scaled)

    # ── Persist ──────────────────────────────────────────────
    def save(self, out_dir: Path | str | None = None):
        out = Path(out_dir or Config.MODELS_DIR)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, out / "scaler.pkl")
        joblib.dump(self.label_encoder, out / "label_encoder.pkl")
        joblib.dump(self.selector, out / "feature_selector.pkl")
        (out / "feature_columns.txt").write_text("\n".join(self.feature_columns))
        (out / "selected_features.txt").write_text("\n".join(self.selected_features))

    def load(self, model_dir: Path | str | None = None):
        d = Path(model_dir or Config.MODELS_DIR)
        self.scaler = joblib.load(d / "scaler.pkl")
        self.label_encoder = joblib.load(d / "label_encoder.pkl")
        self.selector = joblib.load(d / "feature_selector.pkl")
        self.feature_columns = (d / "feature_columns.txt").read_text().strip().split("\n")
        self.selected_features = (d / "selected_features.txt").read_text().strip().split("\n")
