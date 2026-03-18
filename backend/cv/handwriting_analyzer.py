"""
Handwriting feature extraction using computer vision.

Extracts spatial, pressure, and consistency features from handwriting images
to support learning disability screening.
"""
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict


@dataclass
class HandwritingFeatures:
    """Structured features extracted from a handwriting sample."""
    avg_letter_size: float
    line_straightness: float
    letter_spacing: float
    word_spacing: float
    writing_pressure: float
    letter_formation_quality: float
    slant_angle: float
    consistency_score: float
    contour_count: int
    aspect_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ImageQuality:
    """Quality metrics for the input image."""
    is_acceptable: bool
    sharpness: float
    contrast: float
    has_content: bool
    warnings: List[str]


class HandwritingAnalyzer:
    """Extract handwriting features from an image for screening purposes."""

    # ── Image quality thresholds ─────────────────────────────
    MIN_SHARPNESS = 30.0
    MIN_CONTRAST = 20.0
    MIN_CONTOUR_AREA = 50
    MAX_CONTOUR_AREA = 10000

    def __init__(self):
        self.features: HandwritingFeatures | None = None

    # ── Public ───────────────────────────────────────────────
    def check_image_quality(self, image_path: str) -> ImageQuality:
        """Return quality metrics and warnings for the input image."""
        image = cv2.imread(image_path)
        if image is None:
            return ImageQuality(
                is_acceptable=False, sharpness=0, contrast=0,
                has_content=False, warnings=["Could not read image file."],
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = float(gray.std())

        binary = self._binarize(gray)
        contours = self._filter_contours(
            cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        )
        has_content = len(contours) >= 3

        warnings: List[str] = []
        if sharpness < self.MIN_SHARPNESS:
            warnings.append("Image appears blurry. Results may be less reliable.")
        if contrast < self.MIN_CONTRAST:
            warnings.append("Low contrast. Please use dark ink on light paper.")
        if not has_content:
            warnings.append("Very little handwriting detected in the image.")

        return ImageQuality(
            is_acceptable=len(warnings) == 0,
            sharpness=round(sharpness, 2),
            contrast=round(contrast, 2),
            has_content=has_content,
            warnings=warnings,
        )

    def extract_features(self, image_path: str, age: int = 8) -> HandwritingFeatures:
        """Extract all handwriting features from an image."""
        binary, gray = self._preprocess(image_path)
        contours = self._filter_contours(
            cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        )

        if not contours:
            self.features = HandwritingFeatures(
                avg_letter_size=0, line_straightness=0, letter_spacing=0,
                word_spacing=0, writing_pressure=0, letter_formation_quality=0,
                slant_angle=0, consistency_score=0, contour_count=0, aspect_ratio=0,
            )
            return self.features

        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        sorted_boxes = sorted(bounding_boxes, key=lambda b: b[0])

        avg_letter_size = self._avg_letter_size(bounding_boxes)
        line_straightness = self._line_straightness(bounding_boxes)
        letter_spacing = self._letter_spacing(sorted_boxes)
        word_spacing = self._word_spacing(sorted_boxes)
        writing_pressure = self._writing_pressure(gray, contours)
        formation_quality = self._letter_formation_quality(contours)
        slant_angle = self._slant_angle(contours)
        consistency = self._consistency_score(bounding_boxes)
        aspect_ratio = self._avg_aspect_ratio(bounding_boxes)

        self.features = HandwritingFeatures(
            avg_letter_size=round(avg_letter_size, 2),
            line_straightness=round(line_straightness, 2),
            letter_spacing=round(letter_spacing, 2),
            word_spacing=round(word_spacing, 2),
            writing_pressure=round(writing_pressure, 2),
            letter_formation_quality=round(formation_quality, 2),
            slant_angle=round(slant_angle, 2),
            consistency_score=round(consistency, 2),
            contour_count=len(contours),
            aspect_ratio=round(aspect_ratio, 2),
        )
        return self.features

    def calculate_overall_score(self, features: HandwritingFeatures) -> float:
        """Compute a 0-100 overall handwriting quality score."""
        weights = {
            "line_straightness": 0.20,
            "letter_formation_quality": 0.25,
            "consistency_score": 0.20,
            "writing_pressure": 0.15,
            "letter_spacing": 0.10,
            "slant_angle": 0.10,
        }
        score = sum(
            min(100, max(0, getattr(features, k))) * w
            for k, w in weights.items()
        )
        return round(score, 1)

    # ── Preprocessing ────────────────────────────────────────
    def _preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = self._binarize(blurred)

        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return cleaned, gray

    @staticmethod
    def _binarize(gray: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )

    def _filter_contours(self, contours) -> List[np.ndarray]:
        return [
            c for c in contours
            if self.MIN_CONTOUR_AREA < cv2.contourArea(c) < self.MAX_CONTOUR_AREA
        ]

    # ── Feature calculations ─────────────────────────────────
    @staticmethod
    def _avg_letter_size(boxes: list) -> float:
        areas = [w * h for (_, _, w, h) in boxes]
        return float(np.mean(areas)) if areas else 0

    @staticmethod
    def _line_straightness(boxes: list) -> float:
        if len(boxes) < 2:
            return 100.0
        centres_y = [y + h / 2 for (_, y, _, h) in boxes]
        std_y = float(np.std(centres_y))
        mean_h = float(np.mean([h for (_, _, _, h) in boxes]))
        deviation = std_y / (mean_h + 1e-6) * 100
        return max(0, min(100, 100 - deviation))

    @staticmethod
    def _letter_spacing(sorted_boxes: list) -> float:
        if len(sorted_boxes) < 2:
            return 0
        gaps = []
        for i in range(1, len(sorted_boxes)):
            gap = sorted_boxes[i][0] - (sorted_boxes[i - 1][0] + sorted_boxes[i - 1][2])
            if gap > 0:
                gaps.append(gap)
        return float(np.mean(gaps)) if gaps else 0

    @staticmethod
    def _word_spacing(sorted_boxes: list) -> float:
        if len(sorted_boxes) < 2:
            return 0
        gaps = []
        for i in range(1, len(sorted_boxes)):
            gap = sorted_boxes[i][0] - (sorted_boxes[i - 1][0] + sorted_boxes[i - 1][2])
            if gap > 0:
                gaps.append(gap)
        if not gaps:
            return 0
        mean_gap = float(np.mean(gaps))
        large_gaps = [g for g in gaps if g > mean_gap * 1.5]
        return float(np.mean(large_gaps)) if large_gaps else mean_gap * 2

    @staticmethod
    def _writing_pressure(gray: np.ndarray, contours: list) -> float:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)
        ink_pixels = gray[mask > 0]
        if len(ink_pixels) == 0:
            return 0
        mean_intensity = float(np.mean(ink_pixels))
        pressure = max(0, min(100, (255 - mean_intensity) / 255 * 100))
        return pressure

    @staticmethod
    def _letter_formation_quality(contours: list) -> float:
        if not contours:
            return 0
        scores = []
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            scores.append(min(1.0, circularity) * 100)
        return float(np.mean(scores)) if scores else 0

    @staticmethod
    def _slant_angle(contours: list) -> float:
        angles = []
        for c in contours:
            if len(c) >= 5:
                _, (_, _), angle = cv2.fitEllipse(c)
                angles.append(abs(angle - 90))
        return float(np.mean(angles)) if angles else 0

    @staticmethod
    def _consistency_score(boxes: list) -> float:
        if len(boxes) < 2:
            return 100.0
        widths = [w for (_, _, w, h) in boxes]
        heights = [h for (_, _, w, h) in boxes]
        cv_w = float(np.std(widths) / (np.mean(widths) + 1e-6))
        cv_h = float(np.std(heights) / (np.mean(heights) + 1e-6))
        consistency = max(0, 100 - (cv_w + cv_h) * 50)
        return consistency

    @staticmethod
    def _avg_aspect_ratio(boxes: list) -> float:
        ratios = [w / (h + 1e-6) for (_, _, w, h) in boxes]
        return float(np.mean(ratios)) if ratios else 0
