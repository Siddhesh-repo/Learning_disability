"""
Model explainability module.

Provides per-prediction explanations by identifying the top contributing
features and translating them into human-readable risk indicators.
Includes optional SHAP waterfall plots rendered as base64 images.
"""
import base64
import io
import logging
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)

# Human-readable descriptions for each feature
FEATURE_DESCRIPTIONS = {
    "avg_letter_size": "Average letter size",
    "line_straightness": "Line straightness",
    "letter_spacing": "Letter spacing consistency",
    "word_spacing": "Word spacing",
    "writing_pressure": "Writing pressure",
    "letter_formation_quality": "Letter formation quality",
    "slant_angle": "Writing slant angle",
    "consistency_score": "Overall writing consistency",
    "contour_count": "Number of recognisable characters",
    "aspect_ratio": "Character aspect ratio",
    "reading_speed_wpm": "Reading speed (words/min)",
    "pause_frequency": "Pause frequency",
    "average_pause_duration": "Average pause duration",
    "pronunciation_score": "Pronunciation accuracy",
    "fluency_score": "Speech fluency",
    "volume_consistency": "Volume consistency",
    "pitch_variation": "Pitch variation",
    "speech_clarity": "Speech clarity",
    "word_count": "Word count",
    "total_duration": "Total recording duration",
    "age": "Student age",
    "size_per_contour": "Letter size per character",
    "spacing_ratio": "Word-to-letter spacing ratio",
    "quality_consistency": "Quality-consistency product",
    "reading_efficiency": "Reading efficiency",
    "speech_quality": "Overall speech quality",
    "words_per_second": "Words per second",
    "motor_language_link": "Motor-language correlation",
    "overall_consistency": "Cross-modal consistency",
    "age_adj_pressure": "Age-adjusted writing pressure",
    "age_adj_reading": "Age-adjusted reading speed",
}

# For each feature, define whether "high" or "low" is a risk indicator
RISK_DIRECTION = {
    "avg_letter_size": "high",
    "line_straightness": "low",
    "letter_spacing": "high",
    "word_spacing": "high",
    "writing_pressure": "low",
    "letter_formation_quality": "low",
    "slant_angle": "high",
    "consistency_score": "low",
    "contour_count": "low",
    "reading_speed_wpm": "low",
    "pause_frequency": "high",
    "average_pause_duration": "high",
    "pronunciation_score": "low",
    "fluency_score": "low",
    "volume_consistency": "low",
    "pitch_variation": "low",
    "speech_clarity": "low",
    "reading_efficiency": "low",
    "speech_quality": "low",
    "quality_consistency": "low",
    "overall_consistency": "low",
    "motor_language_link": "low",
}


def _generate_shap_plot(model, X_sample, feature_names, predicted_class_idx):
    """
    Generate a SHAP waterfall plot for the given prediction.

    Returns a base64-encoded PNG string, or None if SHAP is unavailable.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use TreeExplainer for tree-based models, KernelExplainer as fallback
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            # Fallback — use a small background for KernelExplainer
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_sample)

        # shap_values may be a list (one per class) or a single array
        if isinstance(shap_values, list):
            sv = shap_values[predicted_class_idx]
        else:
            sv = shap_values

        # Build an Explanation object for the waterfall plot
        explanation = shap.Explanation(
            values=sv[0] if sv.ndim > 1 else sv,
            base_values=explainer.expected_value[predicted_class_idx]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value,
            data=X_sample[0] if X_sample.ndim > 1 else X_sample,
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    except ImportError:
        logger.debug("SHAP not installed — skipping SHAP plot")
        return None
    except Exception:
        logger.debug("SHAP plot generation failed", exc_info=True)
        return None


def explain_prediction(
    feature_values: Dict[str, float],
    feature_importances: List[Dict] | None,
    predicted_condition: str,
    confidence: float,
    top_n: int = 5,
    model=None,
    X_scaled=None,
    feature_names: list | None = None,
    predicted_class_idx: int = 0,
) -> Dict:
    """
    Generate a human-readable explanation for a screening prediction.

    Parameters
    ----------
    feature_values : dict
        Raw feature name -> value mapping for this sample.
    feature_importances : list of dicts or None
        Sorted list from ``predictor.feature_importance()``.
    predicted_condition : str
        The predicted label (normal / dyslexia / dysgraphia).
    confidence : float
        Prediction confidence (0-1).
    top_n : int
        Number of top indicators to return.
    model : sklearn estimator, optional
        Trained model for SHAP explanation.
    X_scaled : ndarray, optional
        Scaled feature vector (1 x n_features) for SHAP.
    feature_names : list, optional
        Feature names corresponding to X_scaled columns.
    predicted_class_idx : int
        Index of the predicted class for SHAP.

    Returns
    -------
    dict with keys: summary, top_indicators, confidence_statement, warnings, shap_plot
    """
    indicators: List[Dict] = []

    if feature_importances:
        for entry in feature_importances[:top_n]:
            fname = entry["feature"]
            importance = entry["importance"]
            value = feature_values.get(fname)
            description = FEATURE_DESCRIPTIONS.get(fname, fname)
            direction = RISK_DIRECTION.get(fname, "unknown")

            indicator = {
                "feature": fname,
                "description": description,
                "value": round(value, 2) if value is not None else None,
                "importance": importance,
                "risk_direction": direction,
            }
            indicators.append(indicator)

    # Confidence statement
    if confidence >= 0.80:
        conf_text = "High confidence"
    elif confidence >= 0.60:
        conf_text = "Moderate confidence"
    else:
        conf_text = "Low confidence — result should be interpreted with caution"

    # Summary
    if predicted_condition == "normal":
        summary = (
            "The screening indicators suggest performance within the normal range. "
            "No immediate risk indicators were identified."
        )
    else:
        condition_label = predicted_condition.replace("_", " ").title()
        summary = (
            f"The screening identifies potential risk indicators for {condition_label}. "
            f"This is a screening result, not a diagnosis. "
            f"Please consult a qualified professional for a comprehensive assessment."
        )

    # Warnings
    warnings = []
    if confidence < 0.50:
        warnings.append(
            "Prediction confidence is below 50%. The input data may be "
            "insufficient or ambiguous for reliable screening."
        )

    # SHAP plot (optional)
    shap_plot = None
    if model is not None and X_scaled is not None and feature_names is not None:
        shap_plot = _generate_shap_plot(model, X_scaled, feature_names, predicted_class_idx)

    return {
        "summary": summary,
        "top_indicators": indicators,
        "confidence_statement": conf_text,
        "warnings": warnings,
        "shap_plot": shap_plot,
    }
