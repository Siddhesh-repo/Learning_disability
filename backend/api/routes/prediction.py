"""
Prediction endpoints for all phases (legacy, phase 2/3/4).
"""
import os
import uuid
import logging
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify

from config import Config
from api.schemas import (
    validate_predict_request,
    validate_file_extension,
    validate_speech_predict_request,
    validate_fusion_predict_request,
)
from cv.handwriting_analyzer import HandwritingAnalyzer
from ml.explainability import explain_prediction
from api.routes.model_service import model_service
from api.routes.auth import token_required
from models.database import Student

logger = logging.getLogger(__name__)

prediction_bp = Blueprint("prediction", __name__)

handwriting_analyzer = HandwritingAnalyzer()


def _to_native(val):
    """Convert numpy types to Python builtins for JSON serialisation."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _save_screening(student_id: int, result_data: dict, feature_dict: dict, phase_predictions: dict = None):
    """Persist screening result to database."""
    try:
        from models.database import db, ScreeningResult
        sr = ScreeningResult(
            student_id=student_id,
            predicted_condition=result_data.get("prediction", {}).get("condition", ""),
            confidence=result_data.get("prediction", {}).get("confidence", 0.0),
            probabilities=result_data.get("prediction", {}).get("probabilities", {}),
            explanation_summary=result_data.get("explanation", {}).get("summary", ""),
            severity_level=result_data.get("recommendations", {}).get("severity_level", ""),
            features=feature_dict,
            phase_predictions=phase_predictions or {}
        )
        db.session.add(sr)
        db.session.commit()
        logger.info("Screening result saved: id=%s", sr.id)
    except Exception:
        logger.debug("Could not save screening (database may not be configured)", exc_info=True)


# ── Legacy Prediction ────────────────────────────────────
@prediction_bp.route("/predict", methods=["POST"])
@token_required
def predict(current_user):
    ms = model_service
    if ms.default.predictor is None:
        return jsonify({
            "error": "ML models not loaded. Run: python train.py",
        }), 503

    data = request.get_json(silent=True)
    ok, msg = validate_predict_request(data)
    if not ok:
        return jsonify({"error": msg}), 400

    student_id = data.get("student_id")
    if not student_id:
        return jsonify({"error": "student_id is required."}), 400
        
    student = Student.query.filter_by(id=student_id, user_id=current_user.id).first()
    if not student:
        return jsonify({"error": "Student not found or access denied."}), 404

    hw = data.get("handwriting_features", {})
    sp = data.get("speech_features", {})
    age = student.age or 8

    try:
        feature_dict = {**hw, **sp, "age": age}
        input_df = pd.DataFrame([feature_dict])
        X = ms.default.engineer.transform(input_df)

        pred_idx = ms.default.predictor.predict(X)[0]
        proba = ms.default.predictor.predict_proba(X)[0]

        label = ms.default.label_encoder.inverse_transform([pred_idx])[0]
        prob_dict = {
            lbl: round(float(p), 4)
            for lbl, p in zip(ms.default.label_encoder.classes_, proba)
        }
        confidence = float(max(proba))

        recs = ms.recommender.generate(label, prob_dict, age)

        imp = ms.default.predictor.feature_importance(ms.default.engineer.selected_features)
        explanation = explain_prediction(
            feature_values=feature_dict,
            feature_importances=imp,
            predicted_condition=label,
            confidence=confidence,
            model=ms.default.predictor.model,
            X_scaled=X,
            feature_names=ms.default.engineer.selected_features,
            predicted_class_idx=int(pred_idx),
        )

        result = {
            "student_id": student.id,
            "student_name": student.name,
            "age": age,
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "condition": label,
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
            },
            "explanation": explanation,
            "recommendations": recs,
        }

        native_features = {k: _to_native(v) for k, v in feature_dict.items()}
        _save_screening(student.id, result, native_features)
        return jsonify(result)

    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


# ── Phase 2: Handwriting ─────────────────────────────────
@prediction_bp.route("/predict/handwriting-phase2", methods=["POST"])
@token_required
def predict_handwriting_phase2(current_user):
    ms = model_service
    if ms.phase2.predictor is None:
        return jsonify({
            "error": "Phase 2 handwriting model not loaded. Run: python train_phase2_handwriting.py",
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error": "student_id is required."}), 400
        
    student = Student.query.filter_by(id=student_id, user_id=current_user.id).first()
    if not student:
        return jsonify({"error": "Student not found or access denied."}), 404

    file = request.files["image"]
    ok, msg = validate_file_extension(file.filename or "", Config.ALLOWED_IMAGE_EXTENSIONS)
    if not ok:
        return jsonify({"error": msg}), 400

    age = student.age or 8

    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.png")
    file.save(tmp)

    try:
        quality = handwriting_analyzer.check_image_quality(tmp)
        features = handwriting_analyzer.extract_features(tmp)
        overall_score = handwriting_analyzer.calculate_overall_score(features)

        feature_dict = {
            **features.to_dict(),
            "age": age,
            "handwriting_overall_score": overall_score,
            "image_sharpness": quality.sharpness,
            "image_contrast": quality.contrast,
        }
        X = ms.phase2.engineer.transform(pd.DataFrame([feature_dict]))

        pred_idx = ms.phase2.predictor.predict(X)[0]
        proba = ms.phase2.predictor.predict_proba(X)[0]
        label = ms.phase2.label_encoder.inverse_transform([pred_idx])[0]

        probabilities = {
            lbl: round(float(p), 4)
            for lbl, p in zip(ms.phase2.label_encoder.classes_, proba)
        }
        confidence = float(max(proba))

        imp = ms.phase2.predictor.feature_importance(ms.phase2.engineer.selected_features)
        explanation = explain_prediction(
            feature_values=feature_dict,
            feature_importances=imp,
            predicted_condition=label,
            confidence=confidence,
        )

        recs = ms.recommender.generate(label, probabilities, age) if ms.recommender else {}

        result = {
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "condition": label,
                "confidence": round(confidence, 4),
                "probabilities": probabilities,
            },
            "features": {k: _to_native(v) for k, v in feature_dict.items()},
            "quality": {
                "is_acceptable": quality.is_acceptable,
                "warnings": quality.warnings,
            },
            "explanation": explanation,
            "recommendations": recs,
            "student_id": student.id,
        }
        
        # Don't natively save phase 2 explicitly to db, 
        # unless you want to track partial steps.
        # But we can if we want to. For now we will return it.
        return jsonify(result)
    except Exception as exc:
        logger.exception("Phase 2 handwriting prediction failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ── Phase 3: Speech ──────────────────────────────────────
@prediction_bp.route("/predict/speech-phase3", methods=["POST"])
@token_required
def predict_speech_phase3(current_user):
    ms = model_service
    if ms.phase3.predictor is None:
        return jsonify({
            "error": "Phase 3 speech model not loaded. Run: python train_phase3_speech.py",
        }), 503

    data = request.get_json(silent=True)
    ok, msg = validate_speech_predict_request(data)
    if not ok:
        return jsonify({"error": msg}), 400

    student_id = data.get("student_id")
    if not student_id:
        return jsonify({"error": "student_id is required."}), 400
        
    student = Student.query.filter_by(id=student_id, user_id=current_user.id).first()
    if not student:
        return jsonify({"error": "Student not found or access denied."}), 404

    age = student.age or 8
    sp = data.get("speech_features", {})

    try:
        feature_dict = {**sp, "age": age}
        X = ms.phase3.engineer.transform(pd.DataFrame([feature_dict]))

        pred_idx = ms.phase3.predictor.predict(X)[0]
        proba = ms.phase3.predictor.predict_proba(X)[0]
        label = ms.phase3.label_encoder.inverse_transform([pred_idx])[0]

        probabilities = {
            lbl: round(float(p), 4)
            for lbl, p in zip(ms.phase3.label_encoder.classes_, proba)
        }
        confidence = float(max(proba))

        imp = ms.phase3.predictor.feature_importance(ms.phase3.engineer.selected_features)
        explanation = explain_prediction(
            feature_values=feature_dict,
            feature_importances=imp,
            predicted_condition=label,
            confidence=confidence,
        )
        recs = ms.recommender.generate(label, probabilities, age) if ms.recommender else {}

        return jsonify({
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "condition": label,
                "confidence": round(confidence, 4),
                "probabilities": probabilities,
            },
            "features": {k: _to_native(v) for k, v in feature_dict.items()},
            "explanation": explanation,
            "recommendations": recs,
            "student_id": student.id,
        })
    except Exception as exc:
        logger.exception("Phase 3 speech prediction failed")
        return jsonify({"error": str(exc)}), 500


# ── Phase 4: Fusion ──────────────────────────────────────
@prediction_bp.route("/predict/fusion-phase4", methods=["POST"])
@token_required
def predict_fusion_phase4(current_user):
    ms = model_service
    if ms.phase4.predictor is None:
        return jsonify({
            "error": "Phase 4 fusion model not loaded. Run: python train_phase4_fusion.py",
        }), 503

    data = request.get_json(silent=True)
    ok, msg = validate_fusion_predict_request(data)
    if not ok:
        return jsonify({"error": msg}), 400

    student_id = data.get("student_id")
    if not student_id:
        return jsonify({"error": "student_id is required."}), 400
        
    student = Student.query.filter_by(id=student_id, user_id=current_user.id).first()
    if not student:
        return jsonify({"error": "Student not found or access denied."}), 404

    hw = data.get("handwriting_features", {})
    sp = data.get("speech_features", {})
    age = student.age or 8
    hw_w = float(data.get("handwriting_weight", 0.5))
    sp_w = float(data.get("speech_weight", 0.5))

    weight_sum = hw_w + sp_w
    hw_w = hw_w / weight_sum
    sp_w = sp_w / weight_sum

    try:
        feature_dict = {**hw, **sp, "age": age}
        X = ms.phase4.engineer.transform(pd.DataFrame([feature_dict]))

        pred_idx = ms.phase4.predictor.predict(X)[0]
        proba = ms.phase4.predictor.predict_proba(X)[0]
        label = ms.phase4.label_encoder.inverse_transform([pred_idx])[0]

        probabilities = {
            lbl: round(float(p), 4)
            for lbl, p in zip(ms.phase4.label_encoder.classes_, proba)
        }
        confidence = float(max(proba))

        imp = ms.phase4.predictor.feature_importance(ms.phase4.engineer.selected_features)
        explanation = explain_prediction(
            feature_values=feature_dict,
            feature_importances=imp,
            predicted_condition=label,
            confidence=confidence,
        )
        recs = ms.recommender.generate(label, probabilities, age) if ms.recommender else {}

        result = {
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "fusion": {
                "handwriting_weight": round(hw_w, 4),
                "speech_weight": round(sp_w, 4),
            },
            "prediction": {
                "condition": label,
                "confidence": round(confidence, 4),
                "probabilities": probabilities,
            },
            "features": {
                "handwriting_features": {k: _to_native(v) for k, v in hw.items()},
                "speech_features": {k: _to_native(v) for k, v in sp.items()},
                "age": age,
            },
            "explanation": explanation,
            "recommendations": recs,
            "student_id": student.id,
        }

        # Format features nicely for the db
        native_features = {k: _to_native(v) for k, v in feature_dict.items()}
        _save_screening(student.id, result, native_features)

        return jsonify(result)
    except Exception as exc:
        logger.exception("Phase 4 fusion prediction failed")
        return jsonify({"error": str(exc)}), 500
