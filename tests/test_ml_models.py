"""
Test suite for ML models with GPU support following TDD principles.
Tests LOSO cross-validation and proper metrics calculation.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneGroupOut
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.classical import (
    LOSOClassifier,
    evaluate_with_loso,
    compute_metrics_with_ci,
    train_svm,
    train_random_forest,
    train_xgboost_gpu,
    ensemble_predictions,
    ClassicalMLPipeline
)


class TestLOSOValidation:
    """Test Leave-One-Subject-Out validation"""

    @pytest.fixture
    def mock_mci_data(self):
        """Create mock MCI dataset with subject grouping"""
        n_subjects = 20
        n_features = 50
        n_epochs_per_subject = 30

        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_subjects * n_epochs_per_subject,
            n_features=n_features,
            n_informative=20,
            n_redundant=10,
            n_repeated=0,
            n_classes=2,
            class_sep=1.5,
            random_state=42
        )

        # Create subject IDs (ensuring all epochs from same subject stay together)
        subject_ids = np.repeat(np.arange(n_subjects), n_epochs_per_subject)

        # Create DataFrame with features
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['subject_id'] = subject_ids
        df['label'] = y

        return df

    def test_loso_split_integrity(self, mock_mci_data):
        """Test that LOSO preserves subject grouping"""
        X = mock_mci_data.drop(['subject_id', 'label'], axis=1)
        y = mock_mci_data['label']
        groups = mock_mci_data['subject_id']

        loso = LeaveOneGroupOut()
        n_splits = loso.get_n_splits(X, y, groups)

        # Should have as many splits as subjects
        assert n_splits == len(np.unique(groups))

        # Check each split
        for train_idx, test_idx in loso.split(X, y, groups):
            train_subjects = groups.iloc[train_idx].unique()
            test_subjects = groups.iloc[test_idx].unique()

            # Test should have exactly one subject
            assert len(test_subjects) == 1

            # No overlap between train and test subjects
            assert len(set(train_subjects) & set(test_subjects)) == 0

    def test_loso_classifier(self, mock_mci_data):
        """Test LOSOClassifier wrapper"""
        X = mock_mci_data.drop(['subject_id', 'label'], axis=1)
        y = mock_mci_data['label']
        groups = mock_mci_data['subject_id']

        clf = LOSOClassifier(base_estimator='svm')
        scores = clf.cross_validate(X, y, groups)

        assert 'f1' in scores
        assert 'mcc' in scores
        assert 'auc' in scores

        # All metrics should be between 0 and 1
        assert 0 <= scores['f1'] <= 1
        assert -1 <= scores['mcc'] <= 1
        assert 0 <= scores['auc'] <= 1


class TestMetrics:
    """Test metric computation with confidence intervals"""

    def test_compute_metrics_with_ci(self):
        """Test metrics calculation with 95% CI"""
        # Create mock predictions with known properties
        n_folds = 10
        y_trues = []
        y_preds = []
        y_probs = []

        for _ in range(n_folds):
            # Generate fold predictions
            y_true = np.array([0, 0, 0, 1, 1, 1])
            # Good predictions with some errors
            y_pred = np.array([0, 0, 1, 1, 1, 0])
            y_prob = np.array([0.2, 0.3, 0.6, 0.7, 0.8, 0.4])

            y_trues.append(y_true)
            y_preds.append(y_pred)
            y_probs.append(y_prob)

        metrics = compute_metrics_with_ci(y_trues, y_preds, y_probs)

        # Check all required metrics exist
        assert 'f1_mean' in metrics
        assert 'f1_ci_lower' in metrics
        assert 'f1_ci_upper' in metrics
        assert 'mcc_mean' in metrics
        assert 'auc_mean' in metrics

        # Check CI makes sense
        assert metrics['f1_ci_lower'] <= metrics['f1_mean'] <= metrics['f1_ci_upper']
        assert metrics['mcc_ci_lower'] <= metrics['mcc_mean'] <= metrics['mcc_ci_upper']
        assert metrics['auc_ci_lower'] <= metrics['auc_mean'] <= metrics['auc_ci_upper']


class TestClassicalModels:
    """Test individual ML models"""

    @pytest.fixture
    def train_test_data(self):
        """Create simple train/test split"""
        X_train, y_train = make_classification(
            n_samples=100,
            n_features=20,
            n_classes=2,
            random_state=42
        )
        X_test, y_test = make_classification(
            n_samples=30,
            n_features=20,
            n_classes=2,
            random_state=43
        )
        return X_train, X_test, y_train, y_test

    def test_train_svm(self, train_test_data):
        """Test SVM training"""
        X_train, X_test, y_train, y_test = train_test_data

        model = train_svm(X_train, y_train, kernel='rbf', C=1.0)

        # Should be able to predict
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

        # Should have predict_proba
        y_prob = model.predict_proba(X_test)
        assert y_prob.shape == (len(y_test), 2)

    def test_train_random_forest(self, train_test_data):
        """Test Random Forest training"""
        X_train, X_test, y_train, y_test = train_test_data

        model = train_random_forest(
            X_train, y_train,
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)

        # RF should have feature importance
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X_train.shape[1]

    @pytest.mark.gpu
    def test_train_xgboost_gpu(self, train_test_data):
        """Test XGBoost with GPU support"""
        X_train, X_test, y_train, y_test = train_test_data

        try:
            model = train_xgboost_gpu(
                X_train, y_train,
                use_gpu=True,
                n_estimators=100,
                learning_rate=0.1
            )

            y_pred = model.predict(X_test)
            assert len(y_pred) == len(y_test)

            # Check GPU was attempted
            assert 'tree_method' in model.get_params()

        except Exception as e:
            # GPU might not be available in test environment
            if "GPU" not in str(e):
                raise


class TestEnsemble:
    """Test ensemble methods"""

    @pytest.fixture
    def mock_predictions(self):
        """Create mock predictions from multiple models"""
        n_samples = 100

        # Three models with different strengths
        pred1 = np.random.rand(n_samples)  # Random baseline
        pred2 = np.where(np.arange(n_samples) < 50, 0.2, 0.8)  # Good separator
        pred3 = np.where(np.arange(n_samples) % 2 == 0, 0.3, 0.7)  # Alternating

        predictions = {
            'svm': pred1,
            'rf': pred2,
            'xgb': pred3
        }

        y_true = np.where(np.arange(n_samples) < 50, 0, 1)

        return predictions, y_true

    def test_ensemble_predictions(self, mock_predictions):
        """Test ensemble averaging"""
        predictions, y_true = mock_predictions

        # Test mean ensemble
        ensemble_mean = ensemble_predictions(
            predictions,
            method='mean'
        )
        assert len(ensemble_mean) == len(y_true)
        assert np.all((ensemble_mean >= 0) & (ensemble_mean <= 1))

        # Test weighted ensemble
        weights = {'svm': 0.2, 'rf': 0.5, 'xgb': 0.3}
        ensemble_weighted = ensemble_predictions(
            predictions,
            method='weighted',
            weights=weights
        )
        assert len(ensemble_weighted) == len(y_true)

        # Test voting
        ensemble_vote = ensemble_predictions(
            predictions,
            method='vote',
            threshold=0.5
        )
        assert len(ensemble_vote) == len(y_true)
        assert np.all(np.isin(ensemble_vote, [0, 1]))


class TestMLPipeline:
    """Test complete ML pipeline"""

    @pytest.fixture
    def pipeline_data(self):
        """Create data for pipeline testing"""
        n_subjects = 10
        n_epochs = 20
        n_features = 30

        data = []
        for subj in range(n_subjects):
            for epoch in range(n_epochs):
                features = np.random.randn(n_features)
                label = subj % 2  # Alternating labels
                row = list(features) + [subj, label]
                data.append(row)

        columns = [f'feat_{i}' for i in range(n_features)] + ['subject_id', 'label']
        df = pd.DataFrame(data, columns=columns)

        return df

    def test_pipeline_initialization(self, pipeline_data):
        """Test pipeline initialization"""
        pipeline = ClassicalMLPipeline(
            models=['svm', 'rf'],
            use_gpu=False
        )

        assert 'svm' in pipeline.models
        assert 'rf' in pipeline.models

    def test_pipeline_fit_predict(self, pipeline_data):
        """Test pipeline fit and predict"""
        X = pipeline_data.drop(['subject_id', 'label'], axis=1)
        y = pipeline_data['label']
        groups = pipeline_data['subject_id']

        pipeline = ClassicalMLPipeline(
            models=['svm'],
            use_gpu=False
        )

        # Fit pipeline
        pipeline.fit(X, y, groups)

        # Should have results
        assert hasattr(pipeline, 'results_')
        assert 'metrics' in pipeline.results_

        # Check metrics
        metrics = pipeline.results_['metrics']
        assert 'f1_mean' in metrics
        assert 'mcc_mean' in metrics
        assert 'auc_mean' in metrics

    def test_pipeline_feature_importance(self, pipeline_data):
        """Test feature importance extraction"""
        X = pipeline_data.drop(['subject_id', 'label'], axis=1)
        y = pipeline_data['label']
        groups = pipeline_data['subject_id']

        pipeline = ClassicalMLPipeline(
            models=['rf'],  # RF has feature importance
            use_gpu=False
        )

        pipeline.fit(X, y, groups)
        importance = pipeline.get_feature_importance()

        assert importance is not None
        assert len(importance) == X.shape[1]

    def test_pipeline_save_load(self, pipeline_data, tmp_path):
        """Test pipeline serialization"""
        X = pipeline_data.drop(['subject_id', 'label'], axis=1)
        y = pipeline_data['label']
        groups = pipeline_data['subject_id']

        pipeline = ClassicalMLPipeline(
            models=['svm'],
            use_gpu=False
        )

        pipeline.fit(X, y, groups)

        # Save pipeline
        save_path = tmp_path / 'pipeline.pkl'
        pipeline.save(save_path)
        assert save_path.exists()

        # Load pipeline
        loaded_pipeline = ClassicalMLPipeline.load(save_path)
        assert loaded_pipeline.results_ is not None

        # Should be able to predict
        predictions = loaded_pipeline.predict(X)
        assert len(predictions) == len(y)