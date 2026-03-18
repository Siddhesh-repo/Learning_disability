"""
Handwriting and speech analysis endpoints.
"""
import os
import uuid
import logging
import tempfile

import numpy as np
from flask import Blueprint, request, jsonify

from config import Config
from api.schemas import validate_file_extension
from cv.handwriting_analyzer import HandwritingAnalyzer
from nlp.speech_analyzer import SpeechAnalyzer
from utils.audio_utils import convert_to_wav

logger = logging.getLogger(__name__)

analysis_bp = Blueprint("analysis", __name__)

# Singleton analysers
handwriting_analyzer = HandwritingAnalyzer()
speech_analyzer = SpeechAnalyzer()


def _to_native(val):
    """Convert numpy types to Python builtins for JSON serialisation."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


# ── Handwriting ──────────────────────────────────────────
@analysis_bp.route("/analyze/handwriting", methods=["POST"])
def analyze_handwriting():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    ok, msg = validate_file_extension(
        file.filename or "", Config.ALLOWED_IMAGE_EXTENSIONS)
    if not ok:
        return jsonify({"error": msg}), 400

    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.png")
    file.save(tmp)
    try:
        quality = handwriting_analyzer.check_image_quality(tmp)
        features = handwriting_analyzer.extract_features(tmp)
        score = handwriting_analyzer.calculate_overall_score(features)

        return jsonify({
            "analysis_id": str(uuid.uuid4()),
            "features": {k: _to_native(v) for k, v in features.to_dict().items()},
            "overall_score": round(score, 1),
            "quality": {
                "is_acceptable": quality.is_acceptable,
                "warnings": quality.warnings,
            },
            "status": "success",
        })
    except Exception as exc:
        logger.exception("Handwriting analysis failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ── Speech ───────────────────────────────────────────────
@analysis_bp.route("/analyze/speech", methods=["POST"])
def analyze_speech():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audio"]
    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename or "recording.webm")
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "webm"

    ok, msg = validate_file_extension(filename, Config.ALLOWED_AUDIO_EXTENSIONS)
    if not ok:
        return jsonify({"error": msg}), 400

    reference_text = request.form.get("referenceText", "")

    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.{ext}")
    wav_path = None
    file.save(tmp)

    try:
        analysis_path = tmp
        if ext != "wav":
            wav_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
            convert_to_wav(tmp, wav_path)
            analysis_path = wav_path

        quality = speech_analyzer.check_audio_quality(analysis_path)
        features = speech_analyzer.analyze(analysis_path, reference_text)

        feat_dict = features.to_dict()
        feat_dict.pop("transcript", None)

        return jsonify({
            "analysis_id": str(uuid.uuid4()),
            "features": {k: _to_native(v) for k, v in feat_dict.items()},
            "transcript": features.transcript,
            "overall_score": round(features.confidence_score, 2),
            "quality": {
                "is_acceptable": quality.is_acceptable,
                "warnings": quality.warnings,
            },
            "status": "success",
        })
    except Exception as exc:
        logger.exception("Speech analysis failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        for p in (tmp, wav_path):
            if p and os.path.exists(p):
                os.remove(p)
