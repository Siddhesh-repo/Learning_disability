# Architecture

## System Overview

The Learning Disability Screening System follows a **three-tier architecture**: a React frontend, a Flask REST API, and a machine-learning inference layer backed by scikit-learn models.

```
┌──────────────┐      HTTP / JSON      ┌──────────────────┐
│   React SPA  │ ◄──────────────────► │  Flask REST API   │
│  (Browser)   │   multipart/form     │  (Python 3.10+)   │
└──────────────┘                       └──────┬───────────┘
                                              │
                         ┌────────────────────┼────────────────────┐
                         │                    │                    │
                    ┌────▼─────┐       ┌──────▼──────┐    ┌───────▼──────┐
                    │ CV Module│       │  NLP Module  │    │  ML Pipeline │
                    │ OpenCV   │       │ librosa/STT  │    │ scikit-learn │
                    └────┬─────┘       └──────┬──────┘    └──────┬───────┘
                         │                    │                   │
                         └────────┬───────────┘                   │
                                  │ features                      │
                                  ▼                               ▼
                         ┌───────────────┐              ┌────────────────┐
                         │   Feature     │──────────►   │  Trained Model │
                         │  Engineering  │              │  (.joblib)     │
                         └───────────────┘              └───────┬────────┘
                                                                │
                                                                ▼
                                                   ┌─────────────────────┐
                                                   │  Explainability +   │
                                                   │  Recommendations    │
                                                   └─────────────────────┘
```

## Data Flow

1. **Frontend** collects student info, a handwriting image, and a speech recording.
2. `/api/analyze/handwriting` — image → OpenCV preprocessing → 10 handwriting features + quality report.
3. `/api/analyze/speech` — audio (WebM→WAV) → librosa analysis + Google STT → 12 speech features + transcript.
4. `/api/predict` — combined feature vector → feature engineering (derived features, scaling) → model inference → prediction + explainability + recommendations.

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `cv/handwriting_analyzer.py` | Image preprocessing, contour detection, feature extraction |
| `nlp/speech_analyzer.py` | Audio analysis, transcription, fluency metrics |
| `ml/data_generator.py` | Synthetic data generation for training |
| `ml/feature_engineering.py` | Derived features, scaling, feature selection |
| `ml/disability_predictor.py` | Model training, tuning, inference |
| `ml/recommendation_engine.py` | Intervention lookup by condition/severity/age |
| `ml/explainability.py` | Feature-importance-based explanations |
| `api/routes.py` | REST endpoints, file handling, validation |

## Deployment

- **Docker Compose** runs the Flask backend (gunicorn, 4 workers) and React frontend (nginx) behind a shared bridge network.
- The backend stores models and uploads in Docker volumes.
- `.env.example` lists all configurable environment variables.

## Real Data CSV Sources (Current Workspace)

The project now has usable real-data CSV inputs for handwriting/text-side and audio-side training:

- Handwriting/text-side real features (Phase 2 output): `backend/data/processed/phase2_handwriting_features.csv`
- Speech/audio real features (generated from real audio files): `backend/data/processed/phase3_speech_features_real.csv`

### How the real speech CSV is generated

`backend/ml/real_speech_data_ingestor.py` scans audio files, infers labels from path or explicit mappings, extracts speech features with `SpeechAnalyzer`, and writes a Phase-3-ready CSV.

Example:

```bash
cd backend
./venv/bin/python ml/real_speech_data_ingestor.py \
     --audio-dir ../../disability_ai_engine/samples/speech \
     --label-map "sample_good=normal,sample_poor=dyslexia" \
     --replicate 25 \
     --output-csv data/processed/phase3_speech_features_real.csv
```

### Real-data phase execution example

```bash
cd backend
./venv/bin/python run_all_phases.py \
     --skip-phase2 \
     --phase3-dataset data/processed/phase3_speech_features_real.csv \
     --phase4-synthetic-samples 1000 \
     --model-types random_forest
```

Note: Phase 4 requires handwriting and speech CSVs sharing the same `sample_id` values for fully real multimodal fusion. If aligned IDs are unavailable, synthetic fusion fallback remains the safe default.
