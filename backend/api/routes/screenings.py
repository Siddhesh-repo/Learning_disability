"""
Screening history endpoints.
"""
import logging
from flask import Blueprint, jsonify
from api.routes.auth import token_required
from models.database import db, ScreeningResult, Student

logger = logging.getLogger(__name__)

screenings_bp = Blueprint("screenings", __name__)


@screenings_bp.route("/screenings", methods=["GET"])
@token_required
def list_screenings(current_user):
    """Return all past screening results for the current educator's students."""
    try:
        # Join with Student to only get screenings for this user's students
        results = (
            ScreeningResult.query
            .join(Student)
            .filter(Student.user_id == current_user.id)
            .order_by(ScreeningResult.created_at.desc())
            .limit(100)
            .all()
        )
        return jsonify([r.to_dict() for r in results])
    except Exception as exc:
        logger.exception("Failed to fetch screenings")
        return jsonify({"error": str(exc)}), 500


@screenings_bp.route("/screenings/<int:screening_id>", methods=["GET"])
@token_required
def get_screening(current_user, screening_id: int):
    """Return a single screening result by ID, if it belongs to the user."""
    try:
        result = (
            ScreeningResult.query
            .join(Student)
            .filter(ScreeningResult.id == screening_id, Student.user_id == current_user.id)
            .first()
        )
        if result is None:
            return jsonify({"error": "Screening not found or access denied."}), 404
        return jsonify(result.to_dict())
    except Exception as exc:
        logger.exception("Failed to fetch screening %s", screening_id)
        return jsonify({"error": str(exc)}), 500
