"""
Application configuration loaded from environment variables.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

    # Server
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", 5001))
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Paths
    MODELS_DIR = BASE_DIR / os.getenv("MODELS_DIR", "models")
    DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
    LOG_DIR = BASE_DIR / "logs"

    # Upload limits
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))

    # Audio
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
    AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

    # Allowed file types
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
    ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "ogg", "m4a", "webm"}

    @classmethod
    def init_dirs(cls):
        """Create required directories if they don't exist."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        (cls.MODELS_DIR / "visualizations").mkdir(exist_ok=True)
        (cls.DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
