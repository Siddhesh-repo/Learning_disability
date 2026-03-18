"""
Tests for the handwriting analyser module.
"""
import os
import sys
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cv.handwriting_analyzer import HandwritingAnalyzer, HandwritingFeatures, ImageQuality


@pytest.fixture
def analyzer():
    return HandwritingAnalyzer()


@pytest.fixture
def sample_image(tmp_path):
    """Create a minimal synthetic handwriting image for testing."""
    img = np.ones((200, 400), dtype=np.uint8) * 255
    # Draw some fake "letters"
    cv2.putText(img, "hello world", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    path = str(tmp_path / "sample.png")
    cv2.imwrite(path, img)
    return path


class TestHandwritingAnalyzer:
    def test_extract_features_returns_dataclass(self, analyzer, sample_image):
        features = analyzer.extract_features(sample_image)
        assert isinstance(features, HandwritingFeatures)

    def test_features_are_non_negative(self, analyzer, sample_image):
        f = analyzer.extract_features(sample_image)
        assert f.avg_letter_size >= 0
        assert f.line_straightness >= 0
        assert f.consistency_score >= 0
        assert f.contour_count >= 0

    def test_overall_score_in_range(self, analyzer, sample_image):
        f = analyzer.extract_features(sample_image)
        score = analyzer.calculate_overall_score(f)
        assert 0 <= score <= 100

    def test_invalid_image_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.extract_features("/nonexistent/path.png")

    def test_image_quality_check(self, analyzer, sample_image):
        quality = analyzer.check_image_quality(sample_image)
        assert isinstance(quality, ImageQuality)
        assert isinstance(quality.warnings, list)
        assert quality.sharpness >= 0

    def test_to_dict(self, analyzer, sample_image):
        f = analyzer.extract_features(sample_image)
        d = f.to_dict()
        assert isinstance(d, dict)
        assert "avg_letter_size" in d

    def test_empty_image(self, analyzer, tmp_path):
        """An all-white image should return zero contours gracefully."""
        img = np.ones((100, 100), dtype=np.uint8) * 255
        path = str(tmp_path / "blank.png")
        cv2.imwrite(path, img)
        f = analyzer.extract_features(path)
        assert f.contour_count == 0
