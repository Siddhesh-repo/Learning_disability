"""
Flask application entry point.
Learning Disability Screening System API.
"""
import os
import sys
import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env in project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from config import Config

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
Config.init_dirs()

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Config.LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
    app.config["SECRET_KEY"] = Config.SECRET_KEY

    # CORS – allow the React frontend origin
    CORS(app, origins=[Config.FRONTEND_URL])

    # Database
    from models.database import init_db
    init_db(app)

    # Register API blueprint (new sub-package structure)
    from api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    logger.info("Flask app created successfully")
    return app


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = create_app()

    print("\n" + "=" * 60)
    print("  LEARNING DISABILITY SCREENING SYSTEM – API")
    print("=" * 60)
    print(f"  Environment : {Config.FLASK_ENV}")
    print(f"  Models dir  : {Config.MODELS_DIR}")
    print(f"  Data dir    : {Config.DATA_DIR}")
    print("=" * 60)
    print("  Endpoints:")
    print("    POST  /api/analyze/handwriting")
    print("    POST  /api/analyze/speech")
    print("    POST  /api/predict")
    print("    POST  /api/predict/handwriting-phase2")
    print("    POST  /api/predict/speech-phase3")
    print("    POST  /api/predict/fusion-phase4")
    print("    GET   /api/screenings")
    print("    GET   /api/model-info")
    print("    GET   /api/health")
    print("=" * 60 + "\n")

    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
    )
