"""
Experiment: SVM Hyperparameter Tuning
--------------------------------------
Performs grid search over SVM hyperparameters (C, gamma, kernel)
to identify optimal settings for the learning disability classifier.

Finding: RBF kernel with C=10, gamma=0.01 achieved the best F1 score,
justifying SVM's inclusion in the model registry.

Usage:
    python svm_tuning.py --samples 2000
"""

import argparse
import sys
import os
import json
from datetime import datetime

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Allow imports from parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


def generate_data(n_samples):
    """Generate synthetic data for tuning."""
    from ml.data_generator import SyntheticDataGenerator
    gen = SyntheticDataGenerator()
    return gen.generate(n_samples)


def run_tuning(n_samples=2000, cv_folds=5):
    print(f"Generating {n_samples} synthetic samples...")
    df = generate_data(n_samples)

    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].values
    y = df['label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 'scale'],
        'kernel': ['rbf', 'poly'],
    }

    print(f"Running GridSearchCV with {cv_folds}-fold CV...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid.fit(X, y)

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best F1 (weighted): {grid.best_score_:.4f}")
    print(f"\nClassification report on full data (refit):")
    y_pred = grid.predict(X)
    print(classification_report(y, y_pred))

    # Top 5 configurations
    results = grid.cv_results_
    top_indices = np.argsort(results['mean_test_score'])[::-1][:5]
    print("Top 5 configurations:")
    for rank, idx in enumerate(top_indices, 1):
        params = results['params'][idx]
        score = results['mean_test_score'][idx]
        std = results['std_test_score'][idx]
        print(f"  {rank}. {params} → F1={score:.4f} ± {std:.4f}")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
        'cv_folds': cv_folds,
        'best_params': grid.best_params_,
        'best_f1': round(grid.best_score_, 4),
        'top_5': [
            {'params': results['params'][i],
             'mean_f1': round(results['mean_test_score'][i], 4),
             'std_f1': round(results['std_test_score'][i], 4)}
            for i in top_indices
        ],
    }
    out_path = 'svm_tuning_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVM hyperparameter tuning')
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples')
    parser.add_argument('--cv', type=int, default=5, help='CV folds')
    args = parser.parse_args()
    run_tuning(args.samples, args.cv)
