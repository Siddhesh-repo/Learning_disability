"""
Request / response validation schemas.

Simple dict-based validation to avoid adding heavy dependencies.
"""
from typing import Dict, List, Tuple


def validate_predict_request(data: dict) -> Tuple[bool, str]:
    """Validate the /predict request payload."""
    if not data:
        return False, "Request body is empty."

    hw = data.get("handwriting_features")
    sp = data.get("speech_features")

    if not hw and not sp:
        return False, "At least one of handwriting_features or speech_features is required."

    REQUIRED_HW = {
        "avg_letter_size", "line_straightness", "letter_spacing",
        "word_spacing", "writing_pressure", "letter_formation_quality",
        "slant_angle", "consistency_score", "contour_count", "aspect_ratio",
    }
    REQUIRED_SP = {
        "reading_speed_wpm", "pause_frequency", "average_pause_duration",
        "pronunciation_score", "fluency_score", "volume_consistency",
        "pitch_variation", "speech_clarity", "word_count", "total_duration",
    }

    if hw:
        missing = REQUIRED_HW - set(hw.keys())
        if missing:
            return False, f"Missing handwriting features: {', '.join(sorted(missing))}"

    if sp:
        missing = REQUIRED_SP - set(sp.keys())
        if missing:
            return False, f"Missing speech features: {', '.join(sorted(missing))}"

    age = data.get("age")
    if age is not None:
        if not isinstance(age, (int, float)) or age < 3 or age > 18:
            return False, "age must be a number between 3 and 18."

    return True, ""


def validate_file_extension(filename: str, allowed: set) -> Tuple[bool, str]:
    """Check that a filename has an allowed extension."""
    if not filename or "." not in filename:
        return False, "Filename must have an extension."
    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in allowed:
        return False, f"Unsupported file type: .{ext}. Allowed: {', '.join(sorted(allowed))}"
    return True, ""


def validate_speech_predict_request(data: dict) -> Tuple[bool, str]:
    """Validate Phase 3 speech-only prediction payload."""
    if not data:
        return False, "Request body is empty."

    sp = data.get("speech_features")
    if not sp:
        return False, "speech_features is required."

    required_sp = {
        "reading_speed_wpm", "pause_frequency", "average_pause_duration",
        "pronunciation_score", "fluency_score", "volume_consistency",
        "pitch_variation", "speech_clarity", "word_count", "total_duration",
    }
    missing = required_sp - set(sp.keys())
    if missing:
        return False, f"Missing speech features: {', '.join(sorted(missing))}"

    age = data.get("age")
    if age is not None and (not isinstance(age, (int, float)) or age < 3 or age > 18):
        return False, "age must be a number between 3 and 18."

    return True, ""


def validate_fusion_predict_request(data: dict) -> Tuple[bool, str]:
    """Validate Phase 4 fusion prediction payload."""
    if not data:
        return False, "Request body is empty."

    hw = data.get("handwriting_features")
    sp = data.get("speech_features")
    if not hw or not sp:
        return False, "Both handwriting_features and speech_features are required."

    # Reuse existing checks via validate_predict_request semantics.
    ok, msg = validate_predict_request(data)
    if not ok:
        return False, msg

    hw_w = data.get("handwriting_weight", 0.5)
    sp_w = data.get("speech_weight", 0.5)
    if not isinstance(hw_w, (int, float)) or not isinstance(sp_w, (int, float)):
        return False, "handwriting_weight and speech_weight must be numeric."
    if hw_w < 0 or sp_w < 0:
        return False, "handwriting_weight and speech_weight must be non-negative."
    if hw_w + sp_w <= 0:
        return False, "At least one modality weight must be greater than zero."

    return True, ""
