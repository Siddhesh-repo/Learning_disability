"""
Centralized model loading service.

Replaces module-level global variables with a proper class that
manages all model-loading lifecycle for each phase.
"""
import logging
from pathlib import Path

from config import Config
from ml.feature_engineering import FeatureEngineer
from ml.disability_predictor import DisabilityPredictor
from ml.recommendation_engine import RecommendationEngine

logger = logging.getLogger(__name__)


class _ModelBundle:
    """Holds a predictor + feature-engineer + label-encoder triplet."""
    __slots__ = ("predictor", "engineer", "label_encoder")

    def __init__(self):
        self.predictor = None
        self.engineer = None
        self.label_encoder = None

    @property
    def ready(self) -> bool:
        return self.predictor is not None


class ModelService:
    """Load and expose all trained model bundles used by the API."""

    def __init__(self):
        self.default = _ModelBundle()
        self.phase2 = _ModelBundle()
        self.phase3 = _ModelBundle()
        self.phase4 = _ModelBundle()
        self.recommender: RecommendationEngine | None = None

    # ── Public queries ───────────────────────────────────────
    @property
    def models_ready(self) -> bool:
        return self.default.ready

    # ── Loading ──────────────────────────────────────────────
    def load_all(self):
        """Attempt to load every model bundle that exists on disk."""
        self._load_default()
        self._load_bundle(self.phase2, Config.MODELS_DIR / "phase2_handwriting", "Phase 2")
        self._load_bundle(self.phase3, Config.MODELS_DIR / "phase3_speech", "Phase 3")
        self._load_bundle(self.phase4, Config.MODELS_DIR / "phase4_fusion", "Phase 4")
        self.recommender = RecommendationEngine()

    # ── Internals ────────────────────────────────────────────
    def _load_default(self):
        models_dir = Config.MODELS_DIR
        if not models_dir.exists():
            logger.warning("Models directory not found: %s", models_dir)
            return
        try:
            eng = FeatureEngineer()
            eng.load(models_dir)
            model_file = self._resolve_model_file(models_dir)
            if model_file is None:
                return
            pred = DisabilityPredictor()
            pred.load(models_dir / model_file)
            self.default.predictor = pred
            self.default.engineer = eng
            self.default.label_encoder = eng.label_encoder
            logger.info("Default model loaded: %s", model_file)
        except Exception:
            logger.exception("Failed to load default models")

    def _load_bundle(self, bundle: _ModelBundle, bundle_dir: Path, name: str):
        if not bundle_dir.exists():
            logger.info("%s models directory not found: %s", name, bundle_dir)
            return
        try:
            eng = FeatureEngineer()
            eng.load(bundle_dir)
            model_file = self._resolve_model_file(bundle_dir)
            if model_file is None:
                logger.warning("No %s model .pkl found in %s", name, bundle_dir)
                return
            pred = DisabilityPredictor()
            pred.load(bundle_dir / model_file)
            bundle.predictor = pred
            bundle.engineer = eng
            bundle.label_encoder = eng.label_encoder
            logger.info("%s model loaded: %s", name, model_file)
        except Exception:
            logger.exception("Failed to load %s model bundle", name)

    @staticmethod
    def _resolve_model_file(directory: Path) -> str | None:
        best_txt = directory / "best_model.txt"
        if best_txt.exists():
            lines = best_txt.read_text().strip().splitlines()
            return lines[1].split(": ", 1)[1].strip()
        pkls = sorted(
            [f for f in directory.iterdir()
             if f.suffix == ".pkl" and "scaler" not in f.name
             and "encoder" not in f.name and "selector" not in f.name],
            key=lambda f: f.stat().st_mtime, reverse=True,
        )
        return pkls[0].name if pkls else None


# Module-level singleton — instantiated once at import time.
model_service = ModelService()
model_service.load_all()
