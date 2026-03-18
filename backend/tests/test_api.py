"""
Integration tests for API endpoints.

Uses Flask test client so no server needs to be running.
"""
import os
import sys
import json
import io
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def sample_image(tmp_path):
    img = np.ones((200, 400), dtype=np.uint8) * 255
    cv2.putText(img, "test text", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    path = str(tmp_path / "test.png")
    cv2.imwrite(path, img)
    return path


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"


class TestHandwritingEndpoint:
    def test_missing_file_returns_400(self, client):
        resp = client.post("/api/analyze/handwriting")
        assert resp.status_code == 400

    def test_valid_image_returns_features(self, client, sample_image):
        with open(sample_image, "rb") as f:
            data = {"image": (f, "test.png")}
            resp = client.post(
                "/api/analyze/handwriting",
                data=data,
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "success"
        assert "features" in body
        assert "quality" in body

    def test_wrong_extension_returns_400(self, client, tmp_path):
        bad = tmp_path / "file.xyz"
        bad.write_text("not an image")
        with open(bad, "rb") as f:
            resp = client.post(
                "/api/analyze/handwriting",
                data={"image": (f, "file.xyz")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 400


class TestPredictEndpoint:
    SAMPLE_PAYLOAD = {
        "student_id": "TEST001",
        "student_name": "Test Student",
        "age": 8,
        "handwriting_features": {
            "avg_letter_size": 1200, "line_straightness": 55,
            "letter_spacing": 28, "word_spacing": 56,
            "writing_pressure": 58, "letter_formation_quality": 52,
            "slant_angle": 12, "consistency_score": 48,
            "contour_count": 65, "aspect_ratio": 1.8,
        },
        "speech_features": {
            "reading_speed_wpm": 45, "pause_frequency": 2.5,
            "average_pause_duration": 1.2, "pronunciation_score": 45,
            "fluency_score": 40, "volume_consistency": 55,
            "pitch_variation": 48, "speech_clarity": 42,
            "word_count": 35, "total_duration": 55,
        },
    }

    def test_missing_body_returns_400(self, client):
        resp = client.post("/api/predict", json={})
        assert resp.status_code == 400

    def test_valid_prediction(self, client):
        resp = client.post("/api/predict", json=self.SAMPLE_PAYLOAD)
        # If models aren't trained, we expect 503
        if resp.status_code == 503:
            pytest.skip("Models not trained yet")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "prediction" in body
        assert body["prediction"]["condition"] in ("normal", "dyslexia", "dysgraphia")
        assert "explanation" in body
        assert "recommendations" in body

    def test_missing_features_returns_400(self, client):
        resp = client.post("/api/predict", json={"age": 8})
        assert resp.status_code == 400


class TestPhasePredictionEndpoints:
    def test_phase2_missing_image_returns_400_or_503(self, client):
        resp = client.post("/api/predict/handwriting-phase2")
        # 503 when phase2 model bundle is not present, else 400 due to missing image.
        assert resp.status_code in (400, 503)

    def test_phase3_missing_body_returns_400_or_503(self, client):
        resp = client.post("/api/predict/speech-phase3", json={})
        # 503 when phase3 model bundle is not present, else 400 due to bad payload.
        assert resp.status_code in (400, 503)

    def test_phase4_missing_body_returns_400_or_503(self, client):
        resp = client.post("/api/predict/fusion-phase4", json={})
        # 503 when phase4 model bundle is not present, else 400 due to bad payload.
        assert resp.status_code in (400, 503)


class TestScreeningsEndpoint:
    def test_get_screenings_returns_list(self, client):
        resp = client.get("/api/screenings")
        assert resp.status_code == 200
        body = resp.get_json()
        assert isinstance(body, list)

    def test_get_invalid_screening_returns_404(self, client):
        resp = client.get("/api/screenings/999999")
        assert resp.status_code == 404


class TestModelInfoEndpoint:
    def test_get_model_info_returns_200(self, client):
        resp = client.get("/api/model-info")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "models_loaded" in body
        assert "visualizations" in body
