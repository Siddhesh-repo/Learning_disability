"""
Health check endpoint.
"""
import logging
from datetime import datetime
from flask import Blueprint, jsonify

from api.routes.model_service import model_service

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": model_service.default.ready,
        "phase2_loaded": model_service.phase2.ready,
        "phase3_loaded": model_service.phase3.ready,
        "phase4_loaded": model_service.phase4.ready,
        "timestamp": datetime.now().isoformat(),
    })
