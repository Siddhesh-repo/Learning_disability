# Model Card

## Model Details

| Property | Value |
|----------|-------|
| **Model type** | Ensemble (best of RF / GB / SVM / MLP) |
| **Framework** | scikit-learn 1.6 |
| **Task** | Multi-class classification |
| **Classes** | `normal`, `dyslexia`, `dysgraphia` |
| **Input** | 22+ engineered features (tabular) |
| **Output** | Class label + probability vector |

## Intended Use

Screening tool to flag children aged 6–12 who may benefit from further professional
evaluation for dyslexia or dysgraphia. **Not intended for clinical diagnosis.**

## Training Procedure

1. Generate synthetic dataset (default 2 000 samples, 40/30/30 split).
2. Apply feature engineering: derived features, standard scaling, feature selection.
3. Train four model types: Random Forest, Gradient Boosting, SVM (RBF), MLP.
4. Optionally tune hyperparameters via `GridSearchCV` (5-fold CV).
5. Select the model with the highest weighted F1 score on the held-out test set.
6. Save the best model, scaler, and feature metadata to `models/`.

## Evaluation

Metrics computed on a stratified test set (default 20% hold-out):

| Metric | Definition |
|--------|-----------|
| Accuracy | Overall correct predictions |
| Weighted F1 | F1 averaged across classes weighted by support |
| Per-class Precision, Recall, F1 | Via `classification_report` |
| Confusion Matrix | Saved as visualisation |

Typical performance on synthetic data achieves **>90% weighted F1**. However, this
reflects synthetic-data consistency, not real-world clinical accuracy.

## Limitations & Risks

- **Synthetic-only evaluation**: no clinical validation has been performed.
- **Class bias**: the generator's class profiles are approximations; the model may not
  generalise to populations with different demographic or linguistic backgrounds.
- **Feature noise**: real handwriting/speech extraction introduces noise not present in
  synthetic feature vectors, potentially degrading performance.
- **No fairness audit**: the model has not been tested for demographic bias (gender,
  ethnicity, socio-economic status).

## Explainability

The system provides feature-importance-based explanations via the `explainability`
module, which:

- Identifies the top features driving the prediction.
- Maps feature names to human-readable descriptions.
- Provides directional guidance (which features indicate risk).
- Adds confidence statements and warnings for low-confidence predictions.

## Recommendations

- Always present results alongside the disclaimer that this is a screening tool.
- Encourage follow-up with a qualified educational psychologist.
- Re-train and validate on real clinical data before any deployment beyond academic
  demonstration.
