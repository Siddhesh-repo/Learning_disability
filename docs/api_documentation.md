# API Documentation

Base URL: `http://localhost:5001/api`

---

## GET `/health`

Health check endpoint.

**Response** `200`
```json
{
  "status": "healthy",
  "models_loaded": true,
  "phase2_loaded": true,
  "phase3_loaded": true,
  "phase4_loaded": true
}
```

---

## POST `/analyze/handwriting`

Analyze a handwriting image and extract features.

**Request** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | yes | Image file (PNG, JPG, JPEG, BMP, TIFF) |

**Response** `200`
```json
{
  "overall_score": 72.5,
  "quality": {
    "is_valid": true,
    "warnings": []
  },
  "features": {
    "avg_letter_size": 32.1,
    "line_straightness": 0.85,
    "letter_spacing": 12.3,
    "word_spacing": 45.6,
    "writing_pressure": 128.4,
    "letter_formation_quality": 0.73,
    "slant_angle": 5.2,
    "consistency_score": 0.81,
    "contour_count": 42,
    "aspect_ratio": 2.1
  }
}
```

**Errors**
| Code | Reason |
|------|--------|
| 400 | Missing `image` field or unsupported format |
| 422 | Image too small or unreadable |

---

## POST `/analyze/speech`

Analyze a speech recording and extract features.

**Request** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | yes | Audio file (WAV, WebM, MP3, OGG, FLAC) |
| `reference_text` | string | no | Expected text for accuracy scoring |

**Response** `200`
```json
{
  "overall_score": 78.0,
  "transcript": "The quick brown fox...",
  "quality": {
    "is_valid": true,
    "warnings": []
  },
  "features": {
    "reading_speed_wpm": 95.0,
    "pause_frequency": 3.2,
    "average_pause_duration": 0.45,
    "pronunciation_score": 0.82,
    "fluency_score": 0.76,
    "volume_consistency": 0.88,
    "pitch_variation": 45.3,
    "speech_clarity": 0.79,
    "confidence_score": 0.81,
    "total_duration": 12.5,
    "word_count": 12
  }
}
```

**Errors**
| Code | Reason |
|------|--------|
| 400 | Missing `audio` field or unsupported format |
| 422 | Audio too short or unreadable |

---

## POST `/predict`

Run the ML prediction pipeline.

**Request** `application/json`
```json
{
  "handwriting_features": { "avg_letter_size": 32.1, "..." : "..." },
  "speech_features": { "reading_speed_wpm": 95.0, "..." : "..." },
  "age": 8,
  "student_id": "STU_001"
}
```

Required fields: `handwriting_features`, `speech_features`, `age`.

**Response** `200`
```json
{
  "prediction": {
    "condition": "dyslexia",
    "confidence": 0.82,
    "probabilities": {
      "normal": 0.12,
      "dyslexia": 0.82,
      "dysgraphia": 0.06
    }
  },
  "explanation": {
    "summary": "The model identified elevated risk...",
    "top_indicators": [...],
    "confidence_statement": "High confidence (82%)",
    "warnings": []
  },
  "recommendations": {
    "severity_level": "moderate",
    "primary_interventions": [...],
    "classroom_accommodations": [...],
    "home_strategies": [...],
    "disclaimer": "..."
  }
}
```

**Errors**
| Code | Reason |
|------|--------|
| 400 | Missing or invalid fields |
| 500 | Model not trained yet |

---

## POST `/predict/handwriting-phase2`

Run Phase 2 handwriting-only prediction directly from an uploaded image.

**Request** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | yes | Handwriting image |
| `age` | number | no | Student age (default 8) |

**Response** `200`
```json
{
  "prediction": {
    "condition": "dyslexia",
    "confidence": 0.84,
    "probabilities": {
      "normal": 0.16,
      "dyslexia": 0.84
    }
  },
  "features": {
    "avg_letter_size": 1201.2,
    "line_straightness": 55.4,
    "...": "..."
  },
  "quality": {
    "is_acceptable": true,
    "warnings": []
  },
  "explanation": {
    "summary": "...",
    "top_indicators": []
  },
  "recommendations": {}
}
```

**Errors**
| Code | Reason |
|------|--------|
| 400 | Missing image or unsupported format |
| 503 | Phase 2 model artifacts not loaded |
| 500 | Inference failure |

---

## POST `/predict/speech-phase3`

Run Phase 3 speech-only prediction.

**Request** `application/json`
```json
{
  "speech_features": {
    "reading_speed_wpm": 65,
    "pause_frequency": 1.8,
    "average_pause_duration": 0.7,
    "pronunciation_score": 60,
    "fluency_score": 58,
    "volume_consistency": 70,
    "pitch_variation": 62,
    "speech_clarity": 61,
    "word_count": 52,
    "total_duration": 40
  },
  "age": 8
}
```

**Errors**
| Code | Reason |
|------|--------|
| 400 | Missing or invalid `speech_features` payload |
| 503 | Phase 3 model artifacts not loaded |
| 500 | Inference failure |

---

## POST `/predict/fusion-phase4`

Run Phase 4 multimodal fusion prediction.

**Request** `application/json`
```json
{
  "handwriting_features": { "avg_letter_size": 1200, "...": "..." },
  "speech_features": { "reading_speed_wpm": 65, "...": "..." },
  "age": 8,
  "handwriting_weight": 0.6,
  "speech_weight": 0.4
}
```

**Errors**
| Code | Reason |
|------|--------|
| 400 | Missing features, invalid age, or invalid weights |
| 503 | Phase 4 model artifacts not loaded |
| 500 | Inference failure |
