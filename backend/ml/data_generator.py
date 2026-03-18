"""
Synthetic data generator for learning disability screening models.

Generates labelled samples that simulate handwriting + speech feature
distributions for normal, dyslexia, and dysgraphia profiles.
Used for initial model prototyping when real clinical data is limited.
"""
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path

from config import Config


class SyntheticDataGenerator:
    """Generate synthetic multimodal feature datasets."""

    CONDITIONS = ("normal", "dyslexia", "dysgraphia")

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ── Handwriting features ─────────────────────────────────
    def _handwriting(self, condition: str) -> dict:
        r = self.rng
        profiles = {
            "normal": dict(
                avg_letter_size=r.normal(800, 200),
                line_straightness=r.normal(80, 8),
                letter_spacing=r.normal(20, 6),
                word_spacing=r.normal(45, 12),
                writing_pressure=r.normal(70, 10),
                letter_formation_quality=r.normal(75, 10),
                slant_angle=r.normal(5, 4),
                consistency_score=r.normal(78, 10),
                contour_count=int(r.normal(85, 15)),
                aspect_ratio=r.normal(1.4, 0.3),
            ),
            "dyslexia": dict(
                avg_letter_size=r.normal(1200, 400),
                line_straightness=r.normal(55, 12),
                letter_spacing=r.normal(28, 12),
                word_spacing=r.normal(56, 24),
                writing_pressure=r.normal(58, 15),
                letter_formation_quality=r.normal(52, 15),
                slant_angle=r.normal(12, 10),
                consistency_score=r.normal(48, 15),
                contour_count=int(r.normal(65, 18)),
                aspect_ratio=r.normal(1.8, 0.5),
            ),
            "dysgraphia": dict(
                avg_letter_size=r.normal(1800, 600),
                line_straightness=r.normal(35, 15),
                letter_spacing=r.normal(45, 20),
                word_spacing=r.normal(90, 40),
                writing_pressure=r.normal(35, 20),
                letter_formation_quality=r.normal(30, 15),
                slant_angle=r.normal(25, 15),
                consistency_score=r.normal(25, 12),
                contour_count=int(r.normal(45, 15)),
                aspect_ratio=r.normal(2.5, 0.8),
            ),
        }
        return profiles[condition]

    # ── Speech features ──────────────────────────────────────
    def _speech(self, condition: str) -> dict:
        r = self.rng
        profiles = {
            "normal": dict(
                reading_speed_wpm=r.normal(95, 15),
                pause_frequency=r.normal(0.8, 0.3),
                average_pause_duration=r.normal(0.3, 0.1),
                pronunciation_score=r.normal(82, 8),
                fluency_score=r.normal(80, 8),
                volume_consistency=r.normal(78, 8),
                pitch_variation=r.normal(75, 10),
                speech_clarity=r.normal(80, 8),
                word_count=int(r.normal(80, 15)),
                total_duration=r.normal(30, 8),
            ),
            "dyslexia": dict(
                reading_speed_wpm=r.normal(45, 15),
                pause_frequency=r.normal(2.5, 1.0),
                average_pause_duration=r.normal(1.2, 0.5),
                pronunciation_score=r.normal(45, 15),
                fluency_score=r.normal(40, 12),
                volume_consistency=r.normal(55, 12),
                pitch_variation=r.normal(48, 15),
                speech_clarity=r.normal(42, 12),
                word_count=int(r.normal(35, 12)),
                total_duration=r.normal(55, 15),
            ),
            "dysgraphia": dict(
                reading_speed_wpm=r.normal(78, 18),
                pause_frequency=r.normal(0.8, 0.4),
                average_pause_duration=r.normal(0.4, 0.2),
                pronunciation_score=r.normal(72, 12),
                fluency_score=r.normal(70, 12),
                volume_consistency=r.normal(68, 12),
                pitch_variation=r.normal(72, 12),
                speech_clarity=r.normal(70, 12),
                word_count=int(r.normal(65, 15)),
                total_duration=r.normal(35, 10),
            ),
        }
        return profiles[condition]

    # ── Dataset generation ───────────────────────────────────
    def generate(self, n_per_condition: int = 1000) -> pd.DataFrame:
        """Return a balanced dataset with handwriting + speech features."""
        rows = []
        for condition in self.CONDITIONS:
            for i in range(n_per_condition):
                age = int(self.rng.integers(6, 13))
                row = {
                    "sample_id": f"{condition}_{i:04d}",
                    **self._handwriting(condition),
                    **self._speech(condition),
                    "age": age,
                    "condition": condition,
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        # Clip negative values for features that must be non-negative
        numeric = df.select_dtypes(include=[np.number]).columns
        df[numeric] = df[numeric].clip(lower=0)
        return df

    def save(self, df: pd.DataFrame) -> str:
        """Save dataset to CSV and return the file path."""
        out_dir = Config.DATA_DIR / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"synthetic_dataset_{ts}.csv"
        df.to_csv(path, index=False)

        meta = {
            "created": ts,
            "samples": len(df),
            "conditions": dict(df["condition"].value_counts()),
            "features": [c for c in df.columns if c not in ("sample_id", "condition")],
        }
        meta_path = out_dir / f"metadata_{ts}.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        return str(path)
