"""
Tests for the ML pipeline: data generation, feature engineering, predictor.
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.data_generator import SyntheticDataGenerator
from ml.feature_engineering import FeatureEngineer
from ml.disability_predictor import DisabilityPredictor
from ml.recommendation_engine import RecommendationEngine
from ml.explainability import explain_prediction


class TestDataGenerator:
    def test_generates_correct_size(self):
        gen = SyntheticDataGenerator(seed=0)
        df = gen.generate(n_per_condition=50)
        assert len(df) == 150  # 3 conditions * 50

    def test_balanced_classes(self):
        gen = SyntheticDataGenerator(seed=0)
        df = gen.generate(n_per_condition=100)
        counts = df["condition"].value_counts()
        assert set(counts.index) == {"normal", "dyslexia", "dysgraphia"}
        assert all(c == 100 for c in counts)

    def test_no_negative_features(self):
        gen = SyntheticDataGenerator(seed=0)
        df = gen.generate(n_per_condition=50)
        numeric = df.select_dtypes(include=[np.number])
        assert (numeric >= 0).all().all()


class TestFeatureEngineer:
    @pytest.fixture
    def dataset(self):
        return SyntheticDataGenerator(seed=0).generate(n_per_condition=50)

    def test_fit_transform_shape(self, dataset):
        eng = FeatureEngineer()
        X, y, selected = eng.fit_transform(dataset)
        assert X.shape[0] == len(dataset)
        assert X.shape[1] == len(selected)
        assert len(y) == len(dataset)

    def test_transform_single_sample(self, dataset):
        eng = FeatureEngineer()
        eng.fit_transform(dataset)
        sample = dataset.iloc[[0]].drop(columns=["condition", "sample_id"], errors="ignore")
        X = eng.transform(sample)
        assert X.shape[0] == 1

    def test_save_and_load(self, dataset, tmp_path):
        eng = FeatureEngineer()
        eng.fit_transform(dataset)
        eng.save(tmp_path)

        eng2 = FeatureEngineer()
        eng2.load(tmp_path)
        assert eng2.feature_columns == eng.feature_columns
        assert eng2.selected_features == eng.selected_features


class TestDisabilityPredictor:
    @pytest.fixture
    def trained(self):
        gen = SyntheticDataGenerator(seed=0)
        df = gen.generate(n_per_condition=100)
        eng = FeatureEngineer()
        X, y, sel = eng.fit_transform(df)
        pred = DisabilityPredictor(model_type="random_forest")
        pred.train(X, y)
        return pred, X, y, sel

    def test_predict_shape(self, trained):
        pred, X, y, _ = trained
        preds = pred.predict(X)
        assert len(preds) == len(X)

    def test_predict_proba_shape(self, trained):
        pred, X, y, _ = trained
        proba = pred.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_evaluate_returns_metrics(self, trained):
        pred, X, y, _ = trained
        metrics = pred.evaluate(X, y, label_names=["dysgraphia", "dyslexia", "normal"])
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_feature_importance(self, trained):
        pred, _, _, sel = trained
        imp = pred.feature_importance(sel)
        assert imp is not None
        assert len(imp) == len(sel)

    def test_save_and_load(self, trained, tmp_path):
        pred, X, y, _ = trained
        path = pred.save(tmp_path)
        pred2 = DisabilityPredictor()
        pred2.load(path)
        preds2 = pred2.predict(X)
        assert np.array_equal(pred.predict(X), preds2)


class TestRecommendationEngine:
    def test_dyslexia_recommendations(self):
        rec = RecommendationEngine()
        result = rec.generate("dyslexia", {"dyslexia": 0.75, "normal": 0.15, "dysgraphia": 0.10}, age=8)
        assert result["severity_level"] in ("mild", "moderate", "severe")
        assert len(result["primary_interventions"]) > 0
        assert "disclaimer" in result

    def test_normal_returns_none_severity(self):
        rec = RecommendationEngine()
        result = rec.generate("normal", {"normal": 0.90, "dyslexia": 0.05, "dysgraphia": 0.05}, age=8)
        assert result["severity_level"] == "none"


class TestExplainability:
    def test_explanation_structure(self):
        features = {"reading_speed_wpm": 45.0, "consistency_score": 30.0}
        importances = [
            {"feature": "reading_speed_wpm", "importance": 0.35},
            {"feature": "consistency_score", "importance": 0.25},
        ]
        result = explain_prediction(features, importances, "dyslexia", 0.75)
        assert "summary" in result
        assert "top_indicators" in result
        assert len(result["top_indicators"]) == 2
        assert "confidence_statement" in result
