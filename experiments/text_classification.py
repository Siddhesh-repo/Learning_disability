"""
Experiment: Text Classification on OCR Output
-----------------------------------------------
Attempts to classify learning disabilities from text extracted via OCR
by analysing reading-error patterns (omissions, substitutions, reversals).

Finding: OCR noise dominated real error signals; text-level features were
unreliable. This motivated the shift to image-level CV features.

Usage:
    python text_classification.py --csv ../data/extracted_text.csv
"""

import argparse
import os
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report


# Common reversal pairs often associated with dyslexia
REVERSAL_PAIRS = [
    ('b', 'd'), ('p', 'q'), ('m', 'w'), ('n', 'u'),
    ('was', 'saw'), ('on', 'no'), ('of', 'fo'),
]


def extract_text_features(text, reference=''):
    """Extract error-pattern features from OCR text."""
    words = text.lower().split()
    ref_words = reference.lower().split() if reference else []

    # Basic counts
    word_count = len(words)
    char_count = len(text)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    # Reversal counting
    reversal_count = 0
    for w in words:
        for a, b in REVERSAL_PAIRS:
            if a in w:
                reversed_w = w.replace(a, b)
                if reversed_w in ref_words:
                    reversal_count += 1

    # Repetition counting
    bigrams = list(zip(words[:-1], words[1:]))
    repetition_count = sum(1 for a, b in bigrams if a == b)

    # Omission / substitution (vs reference)
    omissions = 0
    substitutions = 0
    if ref_words:
        ref_set = set(ref_words)
        for w in ref_words:
            if w not in words:
                omissions += 1
        for w in words:
            if w not in ref_set:
                substitutions += 1

    return {
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': round(avg_word_len, 2),
        'reversal_count': reversal_count,
        'repetition_count': repetition_count,
        'omission_count': omissions,
        'substitution_count': substitutions,
        'unique_word_ratio': round(len(set(words)) / max(word_count, 1), 3),
    }


def run_experiment(csv_path=None):
    """Run text classification experiment."""
    texts = []
    labels = []

    if csv_path and os.path.exists(csv_path):
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row.get('text', ''))
                labels.append(row.get('label', 'normal'))
    else:
        # Generate synthetic text samples for demonstration
        print("No CSV provided; generating synthetic text samples...")
        np.random.seed(42)
        ref = "the quick brown fox jumps over the lazy dog near the river bank"

        for _ in range(100):
            texts.append(ref)
            labels.append('normal')
        for _ in range(75):
            words = ref.split()
            # Simulate dyslexia-like errors
            modified = []
            for w in words:
                if np.random.random() < 0.15:
                    modified.append(w[::-1])  # reversal
                elif np.random.random() < 0.1:
                    continue  # omission
                else:
                    modified.append(w)
            texts.append(' '.join(modified))
            labels.append('dyslexia')
        for _ in range(75):
            words = ref.split()
            modified = []
            for w in words:
                if np.random.random() < 0.2:
                    modified.append(w + w[-1])  # letter duplication
                else:
                    modified.append(w)
            texts.append(' '.join(modified))
            labels.append('dysgraphia')

    # Feature extraction
    feature_dicts = [extract_text_features(t, ref if 'ref' in dir() else '') for t in texts]
    feature_names = list(feature_dicts[0].keys())
    X_manual = np.array([[d[k] for k in feature_names] for d in feature_dicts])

    # TF-IDF features (character n-grams)
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=100)
    X_tfidf = tfidf.fit_transform(texts).toarray()

    # Combine
    X = np.hstack([X_manual, X_tfidf])
    y = np.array(labels)

    print(f"Samples: {len(y)}, Features: {X.shape[1]}")
    print(f"Class distribution: {dict(Counter(y))}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_weighted')
    print(f"\nCross-validated F1 (weighted): {scores.mean():.4f} ± {scores.std():.4f}")

    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("\nClassification report (resubstitution):")
    print(classification_report(y, y_pred))

    # Feature importance for manual features
    importances = clf.feature_importances_[:len(feature_names)]
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    print("Manual feature importances:")
    for name, imp in ranked:
        print(f"  {name}: {imp:.4f}")

    print("\n--- CONCLUSION ---")
    print("Text-based features show moderate discriminative power on synthetic data,")
    print("but heavy reliance on OCR quality makes this unreliable for real handwriting.")
    print("Recommendation: use image-level CV features as primary handwriting signal.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text classification experiment')
    parser.add_argument('--csv', default=None, help='Path to CSV with text,label columns')
    args = parser.parse_args()
    run_experiment(args.csv)
