"""
Model information endpoint — exposes training metrics and visualizations.
"""
import base64
import logging
from pathlib import Path
from flask import Blueprint, jsonify

from config import Config
from api.routes.model_service import model_service

logger = logging.getLogger(__name__)

model_info_bp = Blueprint("model_info", __name__)


@model_info_bp.route("/model-info", methods=["GET"])
def model_info():
    """Return training metrics and visualization images as base64."""
    models_dir = Config.MODELS_DIR
    viz_dir = models_dir / "visualizations"

    info = {
        "models_loaded": model_service.default.ready,
        "default_model": None,
        "visualizations": {},
    }

    # Read best_model.txt for summary
    best_txt = models_dir / "best_model.txt"
    if best_txt.exists():
        info["default_model"] = best_txt.read_text().strip()

    # Encode visualization PNGs as base64
    if viz_dir.exists():
        for png_file in sorted(viz_dir.glob("*.png")):
            try:
                raw = png_file.read_bytes()
                info["visualizations"][png_file.stem] = base64.b64encode(raw).decode("ascii")
            except Exception:
                logger.debug("Could not read visualization: %s", png_file)

    return jsonify(info)
