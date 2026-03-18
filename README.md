# Learning Disability Screening System

An explainable multimodal AI screening system for early risk detection of dyslexia and dysgraphia using handwriting and speech analysis.

> **Disclaimer**: This system is a screening and decision-support tool only. It does **not** provide medical diagnosis. All results should be reviewed by qualified professionals before any clinical or educational decisions are made.

## Overview

This system combines computer vision (handwriting analysis) and natural language processing (speech analysis) to assess risk indicators for learning disabilities in children aged 6–12. It extracts structured features from both modalities, feeds them into trained ML models, and provides:

- **Risk classification**: Normal, Dyslexia-Risk, Dysgraphia-Risk
- **Confidence scores** with uncertainty indication
- **Top contributing indicators** for explainability
- **Personalized intervention recommendations**
- **Printable assessment reports**

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend                      │
│  Consent → Student Info → Handwriting → Speech →    │
│                    Assessment Report                 │
│  (Featuring Canvas Input, Recharts, PDF Export)      │
└──────────────────────┬──────────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────────┐
│                  Flask Backend                        │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Handwriting  │  │   Speech     │  │  Predict   │ │
│  │ Analyzer     │  │   Analyzer   │  │  Endpoint  │ │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘ │
│         └────────┬───────┘                │        │
│          ┌───────▼────────┐        ┌──────▼──────┐ │
│          │   Feature      │        │  Trained    │ │
│          │   Engineering  │───────>│  ML Model   │ │
│          └────────────────┘        └──────┬──────┘ │
│                                    ┌──────▼──────┐ │
│                                    │ Explain +   │ │
│                                    │ Recommend   │ │
│                                    └──────┬──────┘ │
│                               ┌───────────▼────────┐│
│                               │ SQLite DB (History)││
│                               └────────────────────┘│
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- FFmpeg (for audio conversion)

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
cp ../.env.example ../.env      # Edit with your settings
python train.py                 # Train models
python app.py                   # Start API on port 5001
```

### Frontend Setup
```bash
cd frontend
npm install
npm start                       # Start on port 3000
```

### Docker (Recommended)
```bash
docker-compose up --build
```

### Phase 1 Data Ingestion (Real Handwriting)
```bash
cd backend
source venv/bin/activate

# Ingest local dataset from ../dyslexia/{yes,no}
# (label is inferred from immediate parent folder to avoid path-token mislabeling)
python -m ml.real_data_ingestor \
    --local-dyslexia-dir ../../dyslexia \
    --output-manifest data/processed/handwriting_manifest_fixed.csv

# Verify local class balance in manifest
tail -n +2 data/processed/handwriting_manifest_fixed.csv | cut -d, -f3 | sort | uniq -c

# Optional: Download and ingest Kaggle dataset (replace with actual slug)
python -m ml.real_data_ingestor \
    --kaggle-dataset <owner/dataset-slug> \
    --local-dyslexia-dir ../../dyslexia

# Optional: Download and ingest u5awan from a direct zip URL
python -m ml.real_data_ingestor \
    --u5awan-zip-url <https://.../u5awan-dataset.zip> \
    --local-dyslexia-dir ../../dyslexia
```

### Phase 2 Handwriting Model Training (Real Data)
```bash
cd backend
source venv/bin/activate

# Train Phase 2 handwriting model from labeled Phase 1 manifest
python train_phase2_handwriting.py \
    --manifest data/processed/handwriting_manifest_labeled_phase1.csv

# Optional: fast dry run with fewer samples and fewer model types
python train_phase2_handwriting.py \
    --manifest data/processed/handwriting_manifest_labeled_phase1.csv \
    --max-samples 300 \
    --model-types random_forest svm
```

Phase 2 outputs are stored under:
- `backend/data/processed/phase2_handwriting_features.csv`
- `backend/models/phase2_handwriting/`

Phase 2 inference endpoint:
- `POST /api/predict/handwriting-phase2` (multipart form-data with `image` and optional `age`)

### Phase 3 Speech Model Training (Started)
```bash
cd backend
source venv/bin/activate

# Synthetic bootstrap (default)
python train_phase3_speech.py --synthetic-samples 800

# Or train from a real speech-feature dataset CSV
python train_phase3_speech.py --dataset data/processed/speech_features.csv
```

Phase 3 outputs:
- `backend/data/processed/phase3_speech_features.csv`
- `backend/models/phase3_speech/`

### Phase 4 Multimodal Fusion Training (Started)
```bash
cd backend
source venv/bin/activate

# Synthetic bootstrap (default)
python train_phase4_fusion.py --synthetic-samples 1000

# Or train from prepared real feature datasets
python train_phase4_fusion.py \
    --handwriting-csv data/processed/phase2_handwriting_features.csv \
    --speech-csv data/processed/phase3_speech_features.csv
```

Phase 4 outputs:
- `backend/data/processed/phase4_fusion_features.csv`
- `backend/models/phase4_fusion/`

## Project Structure

```
learning_disability/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── backend/
│   ├── app.py                  # Flask application entry point
│   ├── config.py               # Environment-based configuration
│   ├── train.py                # Model training pipeline
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── api/                    # API routes and validation
│   ├── cv/                     # Computer vision / handwriting
│   ├── nlp/                    # Speech analysis
│   ├── ml/                     # ML models, features, recommendations
│   ├── utils/                  # Audio/image utilities
│   ├── models/                 # Saved trained models
│   ├── data/                   # Datasets
│   └── tests/                  # Test suite
├── frontend/
│   ├── public/
│   └── src/
│       ├── components/         # React UI components
│       └── utils/              # API client
├── experiments/                # Research scripts (OCR, SVM tuning)
└── docs/                       # Architecture, model card, ethics
```

## ML Models Compared

| Model              | Description                          |
|---------------------|--------------------------------------|
| Random Forest       | Ensemble of decision trees           |
| Gradient Boosting   | Sequential boosting classifier       |
| SVM (RBF kernel)    | Support vector machine               |
| MLP Neural Network  | Multi-layer perceptron               |

The best-performing model is automatically selected and saved after training.

## API Endpoints

| Method | Endpoint               | Description                    |
|--------|------------------------|--------------------------------|
| POST   | `/api/analyze/handwriting` | Analyze handwriting image    |
| POST   | `/api/analyze/speech`      | Analyze speech audio         |
| POST   | `/api/predict`             | Get screening prediction     |
| GET    | `/api/health`              | Health check                 |

See [API Documentation](docs/api_documentation.md) for details.

## Key Features

- **Multimodal Analysis**: Combines handwriting and speech features
- **Model Comparison**: Trains and compares 4 ML algorithms (RF, SVM, GBM, MLP)
- **Explainability**: Shows top risk indicators per prediction (SHAP integration)
- **Interactive Canvas**: Draw handwriting inputs directly in the browser
- **Rich Analytics**: Visualized radar charts and probability bars using Recharts
- **Sample Gallery**: Pre-loaded samples for immediate testing and demonstration
- **PDF Export**: Save student assessment reports directly to your local machine
- **Screening History**: Automatically persist results to SQLite for later review
- **Recommendations**: Age-appropriate intervention suggestions
- **Ethics-Aware**: Built-in disclaimers and limitations

## Limitations

- Trained primarily on synthetic data; real-world validation is limited
- Not a substitute for professional clinical assessment
- Performance may vary across languages, accents, and age groups
- Handwriting features are sensitive to image quality and angle

See [Ethics & Limitations](docs/ethics_and_limitations.md) for full details.

## License

This project is developed for academic and research purposes only.
