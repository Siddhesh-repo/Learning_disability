"""
Speech feature extraction for learning disability screening.

Analyses audio recordings of children reading aloud to extract fluency,
pronunciation, and prosody features relevant to dyslexia risk assessment.
"""
import os
import numpy as np
import librosa
import speech_recognition as sr
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class SpeechFeatures:
    """Structured features extracted from a speech sample."""
    transcript: str
    reading_speed_wpm: float
    pause_frequency: float
    average_pause_duration: float
    pronunciation_score: float
    fluency_score: float
    volume_consistency: float
    pitch_variation: float
    speech_clarity: float
    confidence_score: float
    total_duration: float
    word_count: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AudioQuality:
    """Quality metrics for the input audio."""
    is_acceptable: bool
    duration: float
    avg_amplitude: float
    clipping_ratio: float
    warnings: List[str]


class SpeechAnalyzer:
    """Extract speech features from an audio file for screening."""

    MIN_DURATION = 2.0        # seconds
    MIN_AMPLITUDE = 0.005
    CLIPPING_THRESHOLD = 0.99

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.features: SpeechFeatures | None = None

    # ── Public ───────────────────────────────────────────────
    def check_audio_quality(self, audio_path: str) -> AudioQuality:
        """Return quality metrics and warnings for the audio file."""
        warnings: List[str] = []
        try:
            audio, sr_rate = librosa.load(audio_path, sr=16000)
        except Exception:
            return AudioQuality(
                is_acceptable=False, duration=0, avg_amplitude=0,
                clipping_ratio=0, warnings=["Could not load audio file."],
            )

        duration = len(audio) / sr_rate
        avg_amp = float(np.mean(np.abs(audio)))
        clipping = float(np.mean(np.abs(audio) > self.CLIPPING_THRESHOLD))

        if duration < self.MIN_DURATION:
            warnings.append(f"Recording too short ({duration:.1f}s). Please record at least 5 seconds.")
        if avg_amp < self.MIN_AMPLITUDE:
            warnings.append("Audio level very low. Please speak louder or move closer to microphone.")
        if clipping > 0.05:
            warnings.append("Audio clipping detected. Please reduce volume or move back from microphone.")

        return AudioQuality(
            is_acceptable=len(warnings) == 0,
            duration=round(duration, 2),
            avg_amplitude=round(avg_amp, 4),
            clipping_ratio=round(clipping, 4),
            warnings=warnings,
        )

    def analyze(self, audio_path: str, reference_text: str = "") -> SpeechFeatures:
        """Extract all speech features from an audio file."""
        try:
            audio, sr_rate = librosa.load(audio_path, sr=16000)
        except Exception:
            return self._default_features("Could not load audio")

        duration = len(audio) / sr_rate
        if duration < 0.5:
            return self._default_features("Audio too short")

        transcript = self._transcribe(audio_path)
        word_count = len(transcript.split()) if transcript and not transcript.startswith("Error") else 0
        reading_speed = (word_count / duration) * 60 if duration > 0 else 0

        pauses = self._detect_pauses(audio, sr_rate)
        pronunciation = self._pronunciation_score(transcript, reference_text)
        fluency = self._fluency_score(pauses, duration)
        volume_consistency = self._volume_consistency(audio)
        pitch_variation = self._pitch_variation(audio, sr_rate)
        clarity = self._speech_clarity(audio, sr_rate)
        confidence = self._overall_confidence(pronunciation, fluency, clarity)

        self.features = SpeechFeatures(
            transcript=transcript,
            reading_speed_wpm=round(reading_speed, 2),
            pause_frequency=round(pauses["frequency"], 2),
            average_pause_duration=round(pauses["avg_duration"], 2),
            pronunciation_score=round(pronunciation, 2),
            fluency_score=round(fluency, 2),
            volume_consistency=round(volume_consistency, 2),
            pitch_variation=round(pitch_variation, 2),
            speech_clarity=round(clarity, 2),
            confidence_score=round(confidence, 2),
            total_duration=round(duration, 2),
            word_count=word_count,
        )
        return self.features

    # ── Transcription ────────────────────────────────────────
    def _transcribe(self, audio_path: str) -> str:
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                return self.recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error: speech service unavailable ({e})"
        except Exception as e:
            return f"Error: {e}"

    # ── Pause detection ──────────────────────────────────────
    @staticmethod
    def _detect_pauses(audio: np.ndarray, sr_rate: int) -> Dict[str, float]:
        frame_ms = 30
        frame_len = int(sr_rate * frame_ms / 1000)
        energy_threshold = np.mean(audio ** 2) * 0.1

        pauses: List[float] = []
        current_silence = 0

        for i in range(0, len(audio) - frame_len, frame_len):
            energy = np.mean(audio[i:i + frame_len] ** 2)
            if energy <= energy_threshold:
                current_silence += 1
            else:
                if current_silence > 0:
                    dur = current_silence * frame_ms / 1000
                    if dur > 0.1:
                        pauses.append(dur)
                current_silence = 0

        return {
            "frequency": len(pauses),
            "avg_duration": float(np.mean(pauses)) if pauses else 0,
            "total_pause_time": sum(pauses),
        }

    # ── Scoring helpers ──────────────────────────────────────
    @staticmethod
    def _pronunciation_score(transcript: str, reference: str) -> float:
        if not transcript or transcript.startswith("Error") or transcript.startswith("Could not"):
            return 0.0
        if reference:
            ref_words = set(reference.lower().split())
            trans_words = set(transcript.lower().split())
            if ref_words:
                return min(100, len(trans_words & ref_words) / len(ref_words) * 100)
        words = len(transcript.split())
        return min(100, 50 + min(words, 10) * 3)

    @staticmethod
    def _fluency_score(pauses: Dict[str, float], duration: float) -> float:
        if duration == 0:
            return 0.0
        score = 70.0
        rate = pauses["frequency"] / duration
        score += 20 if rate < 0.5 else (-30 if rate > 1.5 else 0)
        score += 10 if pauses["avg_duration"] < 0.5 else (-20 if pauses["avg_duration"] > 1.0 else 0)
        return max(0, min(100, score))

    @staticmethod
    def _volume_consistency(audio: np.ndarray) -> float:
        chunk_size = max(1, len(audio) // 10)
        rms_vals = [
            float(np.sqrt(np.mean(audio[i:i + chunk_size] ** 2)))
            for i in range(0, len(audio) - chunk_size, chunk_size)
        ]
        if not rms_vals:
            return 0.0
        mean_rms = np.mean(rms_vals)
        if mean_rms == 0:
            return 0.0
        return max(0, min(100, 100 - float(np.std(rms_vals) / mean_rms * 100)))

    @staticmethod
    def _pitch_variation(audio: np.ndarray, sr_rate: int) -> float:
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr_rate, threshold=0.1)
            pitch_vals = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                p = pitches[idx, t]
                if p > 0:
                    pitch_vals.append(p)
            if not pitch_vals:
                return 50.0
            ratio = float(np.std(pitch_vals) / (np.mean(pitch_vals) + 1e-6))
            if 0.1 <= ratio <= 0.3:
                return min(100, 70 + ratio * 100)
            elif ratio < 0.1:
                return max(0, 40 + ratio * 300)
            else:
                return max(0, 90 - (ratio - 0.3) * 200)
        except Exception:
            return 50.0

    @staticmethod
    def _speech_clarity(audio: np.ndarray, sr_rate: int) -> float:
        try:
            spec = np.abs(librosa.stft(audio))
            spectral_centroid = librosa.feature.spectral_centroid(S=spec, sr=sr_rate)
            mean_centroid = float(np.mean(spectral_centroid))
            return max(0, min(100, mean_centroid / 40))
        except Exception:
            return 50.0

    @staticmethod
    def _overall_confidence(pronunciation: float, fluency: float, clarity: float) -> float:
        return pronunciation * 0.4 + fluency * 0.35 + clarity * 0.25

    @staticmethod
    def _default_features(reason: str = "Analysis failed") -> SpeechFeatures:
        return SpeechFeatures(
            transcript=reason,
            reading_speed_wpm=0, pause_frequency=0, average_pause_duration=0,
            pronunciation_score=0, fluency_score=0, volume_consistency=0,
            pitch_variation=0, speech_clarity=0, confidence_score=0,
            total_duration=0, word_count=0,
        )
