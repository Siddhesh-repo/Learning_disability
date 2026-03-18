"""
Tests for the speech analyser module.
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nlp.speech_analyzer import SpeechAnalyzer, SpeechFeatures, AudioQuality


@pytest.fixture
def analyzer():
    return SpeechAnalyzer()


@pytest.fixture
def sample_audio(tmp_path):
    """Create a simple sine-wave WAV file for testing."""
    import soundfile as sf
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = str(tmp_path / "sample.wav")
    sf.write(path, audio, sr)
    return path


class TestSpeechAnalyzer:
    def test_analyze_returns_dataclass(self, analyzer, sample_audio):
        features = analyzer.analyze(sample_audio)
        assert isinstance(features, SpeechFeatures)

    def test_duration_positive(self, analyzer, sample_audio):
        f = analyzer.analyze(sample_audio)
        assert f.total_duration > 0

    def test_scores_in_range(self, analyzer, sample_audio):
        f = analyzer.analyze(sample_audio)
        assert 0 <= f.fluency_score <= 100
        assert 0 <= f.volume_consistency <= 100

    def test_audio_quality_check(self, analyzer, sample_audio):
        q = analyzer.check_audio_quality(sample_audio)
        assert isinstance(q, AudioQuality)
        assert q.duration > 0

    def test_invalid_audio_returns_default(self, analyzer):
        f = analyzer.analyze("/nonexistent/file.wav")
        assert f.total_duration == 0

    def test_to_dict(self, analyzer, sample_audio):
        f = analyzer.analyze(sample_audio)
        d = f.to_dict()
        assert isinstance(d, dict)
        assert "reading_speed_wpm" in d
