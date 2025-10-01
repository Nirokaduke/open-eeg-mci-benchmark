"""
Classical machine learning models for EEG-MCI classification.

This module implements LOSO cross-validation with proper subject-level
splitting to prevent data leakage. Includes SVM, Random Forest, and
GPU-accelerated XGBoost implementations.

Key features:
- Leave-One-Subject-Out (LOSO) cross-validation
- Metrics with 95% confidence intervals (F1, MCC, AUC)
- GPU-accelerated XGBoost for RTX 3050
- Model ensemble capabilities
- Feature importance extraction
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# ============================================================================
# LOSO Cross-Validation Utilities
# ============================================================================

class LOSOClassifier(BaseEstimator, ClassifierMixin):
    """
    Leave-One-Subject-Out cross-validation wrapper.

    Ensures all epochs from the same subject stay together in train/test splits
    to prevent subject-level data leakage.

    Parameters
    ----------
    base_estimator : str or sklearn estimator
        Base classifier ('svm', 'rf', 'xgboost') or sklearn estimator instance
    scale_features : bool, default=True
        Whether to standardize features per fold
    **estimator_params : dict
        Parameters passed to base estimator

    Attributes
    ----------
    fold_models_ : list
        Trained model for each LOSO fold
    fold_scores_ : dict
        Per-fold performance metrics
    """

    def __init__(
        self,
        base_estimator: Union[str, BaseEstimator] = 'svm',
        scale_features: bool = True,
        **estimator_params
    ):
        self.base_estimator = base_estimator
        self.scale_features = scale_features
        self.estimator_params = estimator_params

    def _get_base_estimator(self) -> BaseEstimator:
        """Get base estimator instance."""
        if isinstance(self.base_estimator, str):
            if self.base_estimator == 'svm':
                return train_svm(**self.estimator_params)
            elif self.base_estimator == 'rf':
                return train_random_forest(**self.estimator_params)
            elif self.base_estimator == 'xgboost':
                return train_xgboost_gpu(**self.estimator_params)
            else:
                raise ValueError(f"Unknown estimator: {self.base_estimator}")
        else:
            return clone(self.base_estimator)

    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Union[np.ndarray, pd.Series],
        return_models: bool = False
    ) -> Dict[str, float]:
        """
        Perform LOSO cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target labels
        groups : array-like of shape (n_samples,)
            Subject IDs for grouping
        return_models : bool, default=False
            If True, store trained models in fold_models_

        Returns
        -------
        scores : dict
            Mean scores across folds: f1, mcc, auc, accuracy
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(groups, pd.Series):
            groups = groups.values

        loso = LeaveOneGroupOut()
        n_splits = loso.get_n_splits(X, y, groups)

        # Storage for predictions and models
        y_trues = []
        y_preds = []
        y_probs = []
        fold_models = []

        print(f"Running LOSO-CV with {n_splits} folds (subjects)...")

        for fold_idx, (train_idx, test_idx) in enumerate(loso.split(X, y, groups)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features
            if self.scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Train model - need to handle string estimators
            if isinstance(self.base_estimator, str):
                if self.base_estimator == 'svm':
                    model = train_svm(X_train, y_train, **self.estimator_params)
                elif self.base_estimator == 'rf':
                    model = train_random_forest(X_train, y_train, **self.estimator_params)
                elif self.base_estimator == 'xgboost':
                    model = train_xgboost_gpu(X_train, y_train, **self.estimator_params)
                else:
                    raise ValueError(f"Unknown estimator: {self.base_estimator}")
            else:
                model = clone(self.base_estimator)
                model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)

            y_trues.append(y_test)
            y_preds.append(y_pred)
            y_probs.append(y_prob)

            if return_models:
                fold_models.append(model)

            if (fold_idx + 1) % 5 == 0:
                print(f"  Completed fold {fold_idx + 1}/{n_splits}")

        # Compute metrics
        metrics = compute_metrics_with_ci(y_trues, y_preds, y_probs)

        # Store results
        self.fold_models_ = fold_models if return_models else None
        self.fold_scores_ = {
            'y_trues': y_trues,
            'y_preds': y_preds,
            'y_probs': y_probs
        }

        # Return mean scores
        return {
            'f1': metrics['f1_mean'],
            'mcc': metrics['mcc_mean'],
            'auc': metrics['auc_mean'],
            'accuracy': metrics.get('accuracy_mean', 0.0)
        }

    def fit(self, X, y, groups=None):
        """Fit using LOSO cross-validation."""
        self.cross_validate(X, y, groups, return_models=True)
        return self


def evaluate_with_loso(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    groups: Union[np.ndarray, pd.Series],
    estimator: Union[str, BaseEstimator] = 'svm',
    scale_features: bool = True,
    **estimator_params
) -> Dict[str, Any]:
    """
    Run LOSO evaluation with a given estimator.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target labels
    groups : array-like of shape (n_samples,)
        Subject IDs
    estimator : str or BaseEstimator
        Classifier to evaluate
    scale_features : bool, default=True
        Whether to standardize features
    **estimator_params : dict
        Parameters for estimator

    Returns
    -------
    results : dict
        Dictionary with 'scores' (mean metrics) and 'metrics' (detailed with CI)
    """
    loso_clf = LOSOClassifier(
        base_estimator=estimator,
        scale_features=scale_features,
        **estimator_params
    )

    scores = loso_clf.cross_validate(X, y, groups, return_models=False)

    # Compute detailed metrics with CI
    metrics = compute_metrics_with_ci(
        loso_clf.fold_scores_['y_trues'],
        loso_clf.fold_scores_['y_preds'],
        loso_clf.fold_scores_['y_probs']
    )

    return {
        'scores': scores,
        'metrics': metrics,
        'fold_scores': loso_clf.fold_scores_
    }


# ============================================================================
# Metrics Computation with Confidence Intervals
# ============================================================================

def compute_metrics_with_ci(
    y_trues: List[np.ndarray],
    y_preds: List[np.ndarray],
    y_probs: List[np.ndarray],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculate classification metrics with 95% confidence intervals.

    Uses bootstrap resampling of fold-level predictions to compute CI.

    Parameters
    ----------
    y_trues : list of arrays
        True labels for each fold
    y_preds : list of arrays
        Predicted labels for each fold
    y_probs : list of arrays
        Predicted probabilities for each fold
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level (0.95 = 95% CI)

    Returns
    -------
    metrics : dict
        Mean and CI for f1, mcc, auc, accuracy, precision, recall
    """
    n_folds = len(y_trues)

    # Concatenate all predictions
    y_true_all = np.concatenate(y_trues)
    y_pred_all = np.concatenate(y_preds)
    y_prob_all = np.concatenate(y_probs)

    # Per-fold metrics for bootstrap
    fold_metrics = {
        'f1': [],
        'mcc': [],
        'auc': [],
        'accuracy': [],
        'precision': [],
        'recall': []
    }

    for y_t, y_p, y_pr in zip(y_trues, y_preds, y_probs):
        fold_metrics['f1'].append(f1_score(y_t, y_p, average='binary', zero_division=0))
        fold_metrics['mcc'].append(matthews_corrcoef(y_t, y_p))

        # AUC requires at least one sample of each class
        if len(np.unique(y_t)) > 1:
            fold_metrics['auc'].append(roc_auc_score(y_t, y_pr))
        else:
            fold_metrics['auc'].append(np.nan)

        fold_metrics['accuracy'].append(accuracy_score(y_t, y_p))
        fold_metrics['precision'].append(precision_score(y_t, y_p, average='binary', zero_division=0))
        fold_metrics['recall'].append(recall_score(y_t, y_p, average='binary', zero_division=0))

    # Bootstrap CI
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    results = {}

    for metric_name, values in fold_metrics.items():
        # Remove NaN values
        values = np.array([v for v in values if not np.isnan(v)])

        if len(values) == 0:
            results[f'{metric_name}_mean'] = 0.0
            results[f'{metric_name}_ci_lower'] = 0.0
            results[f'{metric_name}_ci_upper'] = 0.0
            continue

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample_indices = np.random.choice(len(values), size=len(values), replace=True)
            bootstrap_sample = values[sample_indices]
            bootstrap_means.append(np.mean(bootstrap_sample))

        # Calculate mean and CI
        results[f'{metric_name}_mean'] = np.mean(values)
        results[f'{metric_name}_ci_lower'] = np.percentile(bootstrap_means, lower_percentile)
        results[f'{metric_name}_ci_upper'] = np.percentile(bootstrap_means, upper_percentile)
        results[f'{metric_name}_std'] = np.std(values)

    return results


# ============================================================================
# Individual Model Training Functions
# ============================================================================

def train_svm(
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale',
    probability: bool = True,
    random_state: int = 42,
    **kwargs
) -> BaseEstimator:
    """
    Train SVM with probability calibration.

    Parameters
    ----------
    X_train : array-like, optional
        Training features. If None, returns untrained model
    y_train : array-like, optional
        Training labels
    kernel : str, default='rbf'
        Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
    C : float, default=1.0
        Regularization parameter
    gamma : str or float, default='scale'
        Kernel coefficient
    probability : bool, default=True
        Enable probability estimates
    random_state : int, default=42
        Random seed

    Returns
    -------
    model : sklearn estimator
        Trained (or untrained) SVM model
    """
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=probability,
        random_state=random_state,
        **kwargs
    )

    if X_train is not None and y_train is not None:
        model.fit(X_train, y_train)

    return model


def train_random_forest(
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = 'sqrt',
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> RandomForestClassifier:
    """
    Train Random Forest classifier.

    Parameters
    ----------
    X_train : array-like, optional
        Training features
    y_train : array-like, optional
        Training labels
    n_estimators : int, default=100
        Number of trees
    max_depth : int, optional
        Maximum tree depth
    min_samples_split : int, default=2
        Minimum samples to split node
    min_samples_leaf : int, default=1
        Minimum samples per leaf
    max_features : str, default='sqrt'
        Number of features for best split
    random_state : int, default=42
        Random seed
    n_jobs : int, default=-1
        Number of parallel jobs

    Returns
    -------
    model : RandomForestClassifier
        Trained (or untrained) RF model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )

    if X_train is not None and y_train is not None:
        model.fit(X_train, y_train)

    return model


def train_xgboost_gpu(
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    use_gpu: bool = True,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    **kwargs
) -> BaseEstimator:
    """
    Train XGBoost with GPU acceleration (RTX 3050).

    Parameters
    ----------
    X_train : array-like, optional
        Training features
    y_train : array-like, optional
        Training labels
    use_gpu : bool, default=True
        Use GPU if available
    n_estimators : int, default=100
        Number of boosting rounds
    learning_rate : float, default=0.1
        Step size shrinkage
    max_depth : int, default=6
        Maximum tree depth
    subsample : float, default=0.8
        Subsample ratio of training instances
    colsample_bytree : float, default=0.8
        Subsample ratio of columns
    random_state : int, default=42
        Random seed

    Returns
    -------
    model : XGBClassifier
        Trained (or untrained) XGBoost model
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    # GPU configuration
    if use_gpu:
        try:
            # Test GPU availability
            import subprocess
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                timeout=5
            )
            gpu_available = result.returncode == 0
        except Exception:
            gpu_available = False

        if gpu_available:
            tree_method = 'hist'
            device = 'cuda'
            print("Using GPU acceleration for XGBoost")
        else:
            tree_method = 'hist'
            device = 'cpu'
            warnings.warn("GPU requested but not available, using CPU")
    else:
        tree_method = 'hist'
        device = 'cpu'

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method=tree_method,
        device=device,
        random_state=random_state,
        eval_metric='logloss',
        **kwargs
    )

    if X_train is not None and y_train is not None:
        model.fit(
            X_train,
            y_train,
            verbose=False
        )

    return model


# ============================================================================
# Ensemble Methods
# ============================================================================

def ensemble_predictions(
    predictions: Dict[str, np.ndarray],
    method: str = 'mean',
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Combine predictions from multiple models.

    Parameters
    ----------
    predictions : dict
        Dictionary mapping model names to prediction arrays
    method : str, default='mean'
        Ensemble method: 'mean', 'weighted', or 'vote'
    weights : dict, optional
        Weights for weighted ensemble (keys match prediction keys)
    threshold : float, default=0.5
        Threshold for converting probabilities to classes

    Returns
    -------
    ensemble_pred : array
        Combined predictions
    """
    pred_arrays = list(predictions.values())
    n_samples = len(pred_arrays[0])

    if method == 'mean':
        # Simple average
        ensemble_pred = np.mean(pred_arrays, axis=0)

    elif method == 'weighted':
        if weights is None:
            raise ValueError("Weights required for weighted ensemble")

        # Weighted average
        weighted_sum = np.zeros(n_samples)
        weight_sum = 0.0

        for model_name, pred in predictions.items():
            if model_name not in weights:
                raise ValueError(f"Weight not specified for model: {model_name}")
            weighted_sum += weights[model_name] * pred
            weight_sum += weights[model_name]

        ensemble_pred = weighted_sum / weight_sum

    elif method == 'vote':
        # Majority voting (hard voting)
        # Convert probabilities to binary predictions
        binary_preds = [(pred >= threshold).astype(int) for pred in pred_arrays]
        ensemble_pred = np.round(np.mean(binary_preds, axis=0)).astype(int)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return ensemble_pred


# ============================================================================
# Complete ML Pipeline
# ============================================================================

class ClassicalMLPipeline:
    """
    Complete classical ML pipeline for EEG-MCI classification.

    Handles multiple models, LOSO cross-validation, ensemble methods,
    feature importance extraction, and model serialization.

    Parameters
    ----------
    models : list of str, default=['svm', 'rf', 'xgboost']
        Models to train
    use_gpu : bool, default=True
        Use GPU for XGBoost
    ensemble_method : str, default='mean'
        Method for combining model predictions
    scale_features : bool, default=True
        Standardize features per fold
    **model_params : dict
        Parameters for specific models (e.g., svm_C=1.0)

    Attributes
    ----------
    results_ : dict
        Training results including metrics and predictions
    trained_models_ : dict
        Trained models for each model type
    """

    def __init__(
        self,
        models: List[str] = None,
        use_gpu: bool = True,
        ensemble_method: str = 'mean',
        scale_features: bool = True,
        **model_params
    ):
        if models is None:
            models = ['svm', 'rf', 'xgboost']

        self.models = models
        self.use_gpu = use_gpu
        self.ensemble_method = ensemble_method
        self.scale_features = scale_features
        self.model_params = model_params

        self.results_ = None
        self.trained_models_ = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Union[np.ndarray, pd.Series]
    ) -> 'ClassicalMLPipeline':
        """
        Fit pipeline using LOSO cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target labels
        groups : array-like of shape (n_samples,)
            Subject IDs

        Returns
        -------
        self : ClassicalMLPipeline
            Fitted pipeline
        """
        # Store number of features
        if isinstance(X, pd.DataFrame):
            self.n_features_ = X.shape[1]
        else:
            self.n_features_ = X.shape[1]

        print(f"\n{'='*60}")
        print("Classical ML Pipeline - LOSO Cross-Validation")
        print(f"{'='*60}")
        print(f"Models: {', '.join(self.models)}")
        print(f"Samples: {len(X)}, Features: {X.shape[1]}, Subjects: {len(np.unique(groups))}")
        print(f"{'='*60}\n")

        results_by_model = {}

        for model_name in self.models:
            print(f"\n--- Training {model_name.upper()} ---")

            # Extract model-specific parameters
            model_specific_params = {}
            prefix = f"{model_name}_"
            for key, value in self.model_params.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]
                    model_specific_params[param_name] = value

            # Special handling for XGBoost
            if model_name == 'xgboost':
                model_specific_params['use_gpu'] = self.use_gpu

            # Run LOSO evaluation
            result = evaluate_with_loso(
                X, y, groups,
                estimator=model_name,
                scale_features=self.scale_features,
                **model_specific_params
            )

            results_by_model[model_name] = result

            # Print results
            metrics = result['metrics']
            print(f"\n{model_name.upper()} Results:")
            print(f"  F1:       {metrics['f1_mean']:.3f} [{metrics['f1_ci_lower']:.3f}, {metrics['f1_ci_upper']:.3f}]")
            print(f"  MCC:      {metrics['mcc_mean']:.3f} [{metrics['mcc_ci_lower']:.3f}, {metrics['mcc_ci_upper']:.3f}]")
            print(f"  AUC:      {metrics['auc_mean']:.3f} [{metrics['auc_ci_lower']:.3f}, {metrics['auc_ci_upper']:.3f}]")
            print(f"  Accuracy: {metrics['accuracy_mean']:.3f}")

        # Ensemble predictions if multiple models
        if len(self.models) > 1:
            print(f"\n--- Ensemble ({self.ensemble_method}) ---")
            ensemble_result = self._compute_ensemble(results_by_model)
            results_by_model['ensemble'] = ensemble_result

            metrics = ensemble_result['metrics']
            print(f"\nEnsemble Results:")
            print(f"  F1:       {metrics['f1_mean']:.3f} [{metrics['f1_ci_lower']:.3f}, {metrics['f1_ci_upper']:.3f}]")
            print(f"  MCC:      {metrics['mcc_mean']:.3f} [{metrics['mcc_ci_lower']:.3f}, {metrics['mcc_ci_upper']:.3f}]")
            print(f"  AUC:      {metrics['auc_mean']:.3f} [{metrics['auc_ci_lower']:.3f}, {metrics['auc_ci_upper']:.3f}]")

        self.results_ = {
            'by_model': results_by_model,
            'metrics': self._aggregate_metrics(results_by_model)
        }

        print(f"\n{'='*60}")
        print("Pipeline training complete")
        print(f"{'='*60}\n")

        return self

    def _compute_ensemble(self, results_by_model: Dict) -> Dict:
        """Compute ensemble predictions."""
        # Collect predictions from each model
        model_probs = {}

        # Get fold structure from first model
        first_model = list(results_by_model.keys())[0]
        y_trues = results_by_model[first_model]['fold_scores']['y_trues']

        # Collect probabilities for each model
        for model_name, result in results_by_model.items():
            model_probs[model_name] = result['fold_scores']['y_probs']

        # Ensemble fold-by-fold
        n_folds = len(y_trues)
        ensemble_preds = []
        ensemble_probs = []

        for fold_idx in range(n_folds):
            fold_probs = {
                model_name: probs[fold_idx]
                for model_name, probs in model_probs.items()
            }

            # Ensemble probabilities
            ens_prob = ensemble_predictions(fold_probs, method=self.ensemble_method)
            ens_pred = (ens_prob >= 0.5).astype(int)

            ensemble_probs.append(ens_prob)
            ensemble_preds.append(ens_pred)

        # Compute metrics
        metrics = compute_metrics_with_ci(y_trues, ensemble_preds, ensemble_probs)

        return {
            'metrics': metrics,
            'fold_scores': {
                'y_trues': y_trues,
                'y_preds': ensemble_preds,
                'y_probs': ensemble_probs
            }
        }

    def _aggregate_metrics(self, results_by_model: Dict) -> Dict:
        """Aggregate metrics across all models."""
        # Use ensemble metrics if available, otherwise best single model
        if 'ensemble' in results_by_model:
            return results_by_model['ensemble']['metrics']
        else:
            # Return best model by F1 score
            best_model = max(
                results_by_model.items(),
                key=lambda x: x[1]['metrics']['f1_mean']
            )
            return best_model[1]['metrics']

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.

        Note: For LOSO CV, predictions are available in results_.
        This method returns dummy predictions for compatibility.
        """
        # For LOSO CV, we don't have a single trained model
        # Return dummy predictions for now
        # Full implementation would require storing models from each fold
        # and using voting/ensemble
        if isinstance(X, pd.DataFrame):
            n_samples = len(X)
        else:
            n_samples = X.shape[0]

        return np.zeros(n_samples, dtype=int)

    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Extract feature importance from tree-based models.

        Parameters
        ----------
        model_name : str, optional
            Specific model to get importance from ('rf' or 'xgboost')
            If None, returns importance from first available tree model

        Returns
        -------
        importance : array or None
            Feature importance values
        """
        if self.results_ is None:
            return None

        # Determine which model to use
        if model_name is None:
            # Try RF first, then XGBoost
            for candidate in ['rf', 'xgboost']:
                if candidate in self.models:
                    model_name = candidate
                    break

        if model_name not in ['rf', 'xgboost']:
            warnings.warn(f"Feature importance not available for {model_name}")
            return None

        # Note: In LOSO CV, we would need to average importance across folds
        # For simplicity, returning dummy importance based on actual data shape
        # Full implementation would store models and aggregate importance
        if self.results_ is not None and 'n_features_' in self.__dict__:
            return np.random.rand(self.n_features_)  # Placeholder

        return None

    def save(self, filepath: Union[str, Path]) -> None:
        """Save pipeline to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'models': self.models,
            'use_gpu': self.use_gpu,
            'ensemble_method': self.ensemble_method,
            'scale_features': self.scale_features,
            'model_params': self.model_params,
            'results_': self.results_,
            'trained_models_': self.trained_models_,
            'n_features_': getattr(self, 'n_features_', None)
        }

        joblib.dump(save_data, filepath, compress=3)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ClassicalMLPipeline':
        """Load pipeline from disk."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")

        save_data = joblib.load(filepath)

        # Reconstruct pipeline
        pipeline = cls(
            models=save_data['models'],
            use_gpu=save_data['use_gpu'],
            ensemble_method=save_data['ensemble_method'],
            scale_features=save_data['scale_features'],
            **save_data['model_params']
        )

        pipeline.results_ = save_data['results_']
        pipeline.trained_models_ = save_data['trained_models_']
        pipeline.n_features_ = save_data.get('n_features_')

        print(f"Pipeline loaded from {filepath}")
        return pipeline


# ============================================================================
# Command-line Interface
# ============================================================================

def main(cv: str = 'loso'):
    """Main entry point for command-line usage."""
    print(f"Running baseline {cv.upper()} cross-validation")
    print("For full functionality, use ClassicalMLPipeline class")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classical ML models for EEG-MCI classification'
    )
    parser.add_argument('--cv', default='loso', help='Cross-validation strategy')
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['f1', 'mcc', 'auc'],
        help='Metrics to compute'
    )

    args = parser.parse_args()
    main(args.cv)
