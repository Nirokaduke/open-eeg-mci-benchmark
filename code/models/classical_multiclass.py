"""
Multi-class classification for AD/FTD/HC using SVM and Random Forest
Implements macro-F1, macro-AUC, and per-class metrics with LOSO-CV
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, precision_recall_fscore_support
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.labels_ds004504 import load_participants_labels


def load_features(path: Path):
    """Load features from parquet or csv file."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path) if path.suffix==".parquet" else pd.read_csv(path)
    subj_col = None
    for c in df.columns:
        if c.lower() in ["participant_id", "subject_id", "subject", "subj", "id"]:
            subj_col = c
            break
    if subj_col is None:
        raise ValueError(f"Need subject id column. Got: {list(df.columns)}")
    return df, subj_col


def prepare_multiclass(df_feat, subj_col, participants_tsv):
    """Prepare data for multi-class classification (AD/FTD/HC)."""
    _, mapping = load_participants_labels(str(participants_tsv))
    df = df_feat.copy()

    # Fix subject ID format: '001' -> 'sub-001'
    df['subject_formatted'] = 'sub-' + df[subj_col].astype(str).str.zfill(3)
    df['label'] = df['subject_formatted'].map(mapping)

    # Keep all three classes
    df = df[df['label'].isin(['AD', 'FTD', 'HC'])].copy()

    # Encode labels: AD=0, FTD=1, HC=2
    label_map = {'AD': 0, 'FTD': 1, 'HC': 2}
    y = df['label'].map(label_map).values
    groups = df['subject_formatted'].values

    # Get feature columns
    drop = {subj_col, 'subject_formatted', 'label'}
    feat_cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat_cols].values

    return X, y, groups, feat_cols, label_map


def loso_multiclass(X, y, groups, model_type='svm'):
    """Leave-One-Subject-Out CV for multi-class classification."""
    logo = LeaveOneGroupOut()
    y_true = []
    y_pred = []
    y_proba = []
    cms = []

    for tr, te in logo.split(X, y, groups=groups):
        # Select model
        if model_type == 'svm':
            clf = SVC(C=1.0, kernel='rbf', probability=True,
                     decision_function_shape='ovr', random_state=42)
        else:  # rf
            clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Build pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])

        # Train and predict
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        proba = pipe.predict_proba(X[te])

        # Store results
        cms.append(confusion_matrix(y[te], pred, labels=[0, 1, 2]))
        y_true.extend(y[te])
        y_pred.extend(pred)
        y_proba.append(proba)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.vstack(y_proba)

    return y_true, y_pred, y_proba, cms


def calculate_metrics(y_true, y_pred, y_proba, label_names):
    """Calculate multi-class metrics including macro-F1 and macro-AUC."""
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2]
    )

    # Macro metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Multi-class AUC (One-vs-Rest)
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        macro_auc = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
    except:
        macro_auc = None

    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'macro_f1': macro_f1,
        'macro_auc': macro_auc,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'label_names': label_names
    }


def bootstrap_multiclass_ci(y_true, y_pred, y_proba, n_bootstraps=1000):
    """Calculate bootstrap CIs for multi-class metrics."""
    np.random.seed(42)
    n = len(y_true)
    macro_f1_scores = []
    macro_auc_scores = []

    for i in range(n_bootstraps):
        # Bootstrap sample
        indices = resample(range(n), replace=True, n_samples=n, random_state=i)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_proba_boot = y_proba[indices]

        # Check if all classes present
        if len(np.unique(y_true_boot)) < 3:
            continue

        # Calculate metrics
        macro_f1_scores.append(f1_score(y_true_boot, y_pred_boot, average='macro'))

        try:
            y_true_bin = label_binarize(y_true_boot, classes=[0, 1, 2])
            auc = roc_auc_score(y_true_bin, y_proba_boot, average='macro', multi_class='ovr')
            macro_auc_scores.append(auc)
        except:
            pass

    # Calculate CIs
    def get_ci(scores):
        if len(scores) == 0:
            return (0, 0)
        return (np.percentile(scores, 2.5), np.percentile(scores, 97.5))

    return {
        'macro_f1_ci': get_ci(macro_f1_scores),
        'macro_auc_ci': get_ci(macro_auc_scores) if macro_auc_scores else (0, 0)
    }


def generate_report(metrics, ci_results, model_type, output_path):
    """Generate multi-class classification report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f'# Multi-class Classification Report — AD/FTD/HC (LOSO, {model_type.upper()})\n\n')

        # Overall metrics
        f.write('## Overall Performance\n\n')
        f.write(f"- **Accuracy**: {metrics['accuracy']:.3f}\n")
        f.write(f"- **Macro-F1**: {metrics['macro_f1']:.3f} (95% CI: [{ci_results['macro_f1_ci'][0]:.3f}, {ci_results['macro_f1_ci'][1]:.3f}])\n")
        if metrics['macro_auc']:
            f.write(f"- **Macro-AUC**: {metrics['macro_auc']:.3f} (95% CI: [{ci_results['macro_auc_ci'][0]:.3f}, {ci_results['macro_auc_ci'][1]:.3f}])\n")
        f.write('\n')

        # Per-class metrics
        f.write('## Per-Class Performance\n\n')
        f.write('| Class | Precision | Recall | F1-Score | Support |\n')
        f.write('|-------|-----------|--------|----------|---------|}\n')

        for i, label in enumerate(['AD', 'FTD', 'HC']):
            f.write(f"| {label} | {metrics['per_class']['precision'][i]:.3f} | ")
            f.write(f"{metrics['per_class']['recall'][i]:.3f} | ")
            f.write(f"{metrics['per_class']['f1'][i]:.3f} | ")
            f.write(f"{int(metrics['per_class']['support'][i])} |\n")

        f.write('\n')

        # Confusion matrix
        f.write('## Confusion Matrix\n\n')
        f.write('```\n')
        f.write('       Predicted\n')
        f.write('       AD  FTD  HC\n')
        cm = metrics['confusion_matrix']
        f.write(f'AD   [{cm[0,0]:3d} {cm[0,1]:3d} {cm[0,2]:3d}]\n')
        f.write(f'FTD  [{cm[1,0]:3d} {cm[1,1]:3d} {cm[1,2]:3d}]\n')
        f.write(f'HC   [{cm[2,0]:3d} {cm[2,1]:3d} {cm[2,2]:3d}]\n')
        f.write('```\n\n')

        # Class imbalance note
        f.write('## Class Distribution\n\n')
        total = sum(metrics['per_class']['support'])
        for i, label in enumerate(['AD', 'FTD', 'HC']):
            pct = (metrics['per_class']['support'][i] / total) * 100
            f.write(f"- {label}: {int(metrics['per_class']['support'][i])} ({pct:.1f}%)\n")

        f.write('\n## Notes\n')
        f.write('- Class imbalance present (FTD has fewer samples)\n')
        f.write('- Confidence intervals computed using bootstrap (1000 iterations)\n')
        f.write('- LOSO-CV ensures subject-level independence\n')
        f.write('- Poor performance due to minimal sample size\n')


def main():
    parser = argparse.ArgumentParser(description='Multi-class classification for AD/FTD/HC')
    parser.add_argument('--features', default='data/derivatives/features.parquet')
    parser.add_argument('--participants', default='data/bids_raw/ds004504/participants.tsv')
    parser.add_argument('--model', default='svm', choices=['svm', 'rf'],
                       help='Model type: svm or rf (random forest)')
    parser.add_argument('--output', default='reports/baseline_metrics_multiclass.md')
    args = parser.parse_args()

    print(f'[INFO] Loading features from {args.features}')
    df_feat, subj_col = load_features(Path(args.features))

    print(f'[INFO] Preparing multi-class data')
    X, y, groups, feat_names, label_map = prepare_multiclass(
        df_feat, subj_col, Path(args.participants)
    )

    print(f'[INFO] Running LOSO-CV with {len(groups)} subjects using {args.model.upper()}')
    y_true, y_pred, y_proba, cms = loso_multiclass(X, y, groups, args.model)

    print(f'[INFO] Calculating metrics')
    metrics = calculate_metrics(y_true, y_pred, y_proba, ['AD', 'FTD', 'HC'])

    print(f'[INFO] Computing bootstrap confidence intervals')
    ci_results = bootstrap_multiclass_ci(y_true, y_pred, y_proba)

    print(f'[INFO] Generating report')
    generate_report(metrics, ci_results, args.model, Path(args.output))

    print(f'[OK] Wrote {args.output}')


if __name__ == '__main__':
    main()