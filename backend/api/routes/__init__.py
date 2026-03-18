"""
API route sub-package.

Aggregates blueprints from analysis, prediction, health, screenings,
and model_info into a single parent blueprint for backward compatibility.
"""
from flask import Blueprint

from api.routes.health import health_bp
from api.routes.analysis import analysis_bp
from api.routes.prediction import prediction_bp
from api.routes.screenings import screenings_bp
from api.routes.model_info import model_info_bp
from api.routes.auth import auth_bp
from api.routes.students import students_bp

api_bp = Blueprint("api", __name__)

api_bp.register_blueprint(health_bp)
api_bp.register_blueprint(analysis_bp)
api_bp.register_blueprint(prediction_bp)
api_bp.register_blueprint(screenings_bp)
api_bp.register_blueprint(model_info_bp)
api_bp.register_blueprint(auth_bp, url_prefix="/auth")
api_bp.register_blueprint(students_bp, url_prefix="/students")
