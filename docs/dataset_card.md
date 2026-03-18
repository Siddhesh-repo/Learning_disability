# Dataset Card

## Overview

| Property | Value |
|----------|-------|
| **Name** | LD Screening Synthetic Dataset |
| **Version** | 1.0 |
| **Task** | Multi-class classification (normal / dyslexia / dysgraphia) |
| **Modality** | Tabular (derived from image + audio features) |
| **License** | Academic use only |
| **Created** | 2025 |

## Motivation

Clinical datasets for learning disabilities in children are scarce and protected by
privacy regulations. This project uses **synthetic data** generated from published
research statistics to build and validate the screening model. The synthetic generator
produces realistic feature distributions based on reported clinical ranges.

## Data Description

Each sample contains **22+ features** drawn from two modalities:

### Handwriting features (10)
| Feature | Type | Range | Source |
|---------|------|-------|--------|
| avg_letter_size | float | 10–80 px | OpenCV contour analysis |
| line_straightness | float | 0–1 | Linear regression on baselines |
| letter_spacing | float | 5–50 px | Inter-contour distance |
| word_spacing | float | 20–100 px | Gap clustering |
| writing_pressure | float | 50–200 | Mean pixel intensity (inverted) |
| letter_formation_quality | float | 0–1 | Template matching score |
| slant_angle | float | -30–30° | PCA on contour orientation |
| consistency_score | float | 0–1 | CV of letter sizes |
| contour_count | int | 5–200 | Total detected contours |
| aspect_ratio | float | 0.5–5.0 | Bounding-box W/H |

### Speech features (12)
| Feature | Type | Range | Source |
|---------|------|-------|--------|
| reading_speed_wpm | float | 20–200 | word_count / duration |
| pause_frequency | float | 0–20 | Silence intervals per minute |
| average_pause_duration | float | 0.1–3.0 s | Mean silence length |
| pronunciation_score | float | 0–1 | Sequence matching vs. reference |
| fluency_score | float | 0–1 | Composite metric |
| volume_consistency | float | 0–1 | 1 - CV of RMS energy |
| pitch_variation | float | 0–200 Hz | Std of F0 |
| speech_clarity | float | 0–1 | Spectral flatness proxy |
| confidence_score | float | 0–1 | STT API confidence |
| total_duration | float | 1–60 s | Audio length |
| word_count | int | 1–200 | Transcribed word count |

### Derived features (added during feature engineering)
spacing_ratio, quality_consistency, reading_efficiency, speech_quality,
motor_language_link, age_adjusted_reading_speed, age_adjusted_letter_size.

## Generation Process

Profiles for each class (normal, dyslexia, dysgraphia) define per-feature Gaussian
parameters (mean, std) and clipping bounds. Samples are drawn independently with
configurable class balance (default 40/30/30). See `ml/data_generator.py`.

## Limitations

- **Not real clinical data** — model generalisability to real children is unverified.
- Feature ranges are approximations from published literature; actual distributions
  might be multi-modal, skewed, or correlated in ways the generator does not capture.
- No longitudinal or demographic diversity modelling.

## Ethical Considerations

- Synthetic data avoids privacy risks but may introduce systematic biases not present
  in real-world populations.
- Models trained on this data should **never** be used for diagnostic purposes without
  validation on real clinical data and IRB-approved studies.
