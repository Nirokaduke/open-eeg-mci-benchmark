import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import resample
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.labels_ds004504 import load_participants_labels

def load_features(path: Path):
    if not path.exists(): raise FileNotFoundError(path)
    df = pd.read_parquet(path) if path.suffix==".parquet" else pd.read_csv(path)
    subj_col=None
    for c in df.columns:
        if c.lower() in ["participant_id","subject_id","subject","subj","id"]:
            subj_col=c; break
    if subj_col is None: raise ValueError(f"Need subject id column in features. Got: {list(df.columns)}")
    return df, subj_col

def to_binary(df_feat, subj_col, participants_tsv):
    _, mapping = load_participants_labels(str(participants_tsv))
    df = df_feat.copy()
    # Fix subject ID format: '001' -> 'sub-001'
    df['subject_formatted'] = 'sub-' + df[subj_col].astype(str).str.zfill(3)
    df['label'] = df['subject_formatted'].map(mapping)
    df = df[df['label'].isin(['AD','HC'])].copy()
    y = (df['label']=='AD').astype(int).values
    groups = df['subject_formatted'].values  # Use formatted IDs for grouping
    drop={subj_col,'subject_formatted','label'}
    feat_cols=[c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    X=df[feat_cols].values
    return X,y,groups,feat_cols

def loso_svm(X,y,groups):
    logo=LeaveOneGroupOut()
    y_true=[]; y_pred=[]; y_prob=[]; cms=[]
    for tr,te in logo.split(X,y,groups=groups):
        pipe=Pipeline([('scaler',StandardScaler()),('clf',SVC(C=1.0,kernel='rbf',probability=True,random_state=42))])
        pipe.fit(X[tr],y[tr])
        proba=pipe.predict_proba(X[te])[:,1]
        pred=(proba>=0.5).astype(int)
        cms.append(confusion_matrix(y[te],pred,labels=[0,1]))
        y_true+=y[te].tolist(); y_pred+=pred.tolist(); y_prob+=proba.tolist()
    y_true=np.array(y_true); y_pred=np.array(y_pred); y_prob=np.array(y_prob)
    return {
        'F1': float(f1_score(y_true,y_pred)),
        'MCC': float(matthews_corrcoef(y_true,y_pred)),
        'AUC': float(roc_auc_score(y_true,y_prob)),
        'N': int(len(y_true))
    }, cms, y_true, y_pred, y_prob

def bootstrap_ci(y_true, y_pred, y_prob, n_bootstraps=2000, ci=95):
    """Calculate bootstrap confidence intervals for metrics."""
    np.random.seed(42)
    n = len(y_true)
    f1_scores = []
    mcc_scores = []
    auc_scores = []

    for i in range(n_bootstraps):
        # Bootstrap sample with replacement
        indices = resample(range(n), replace=True, n_samples=n, random_state=i)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        # Calculate metrics for bootstrap sample
        f1_scores.append(f1_score(y_true_boot, y_pred_boot))
        mcc_scores.append(matthews_corrcoef(y_true_boot, y_pred_boot))
        try:
            auc_scores.append(roc_auc_score(y_true_boot, y_prob_boot))
        except:
            # AUC undefined if only one class
            pass

    # Calculate confidence intervals
    alpha = (100 - ci) / 2

    def get_ci(scores):
        if len(scores) == 0:
            return (0, 0)
        lower = np.percentile(scores, alpha)
        upper = np.percentile(scores, 100 - alpha)
        return (lower, upper)

    return {
        'F1_CI': get_ci(f1_scores),
        'MCC_CI': get_ci(mcc_scores),
        'AUC_CI': get_ci(auc_scores) if auc_scores else (0, 0)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features', default='data/derivatives/features.parquet')
    ap.add_argument('--participants', default='data/bids_raw/ds004504/participants.tsv')
    ap.add_argument('--bootstrap', type=int, default=2000, help='Number of bootstrap iterations for CI')
    args=ap.parse_args()

    print(f'[INFO] Loading features from {args.features}')
    df_feat, subj_col = load_features(Path(args.features))

    print(f'[INFO] Processing labels from {args.participants}')
    X,y,groups,feat_names = to_binary(df_feat, subj_col, Path(args.participants))

    print(f'[INFO] Running LOSO-CV with {len(groups)} subjects')
    metrics, cms, y_true, y_pred, y_prob = loso_svm(X,y,groups)

    print(f'[INFO] Computing bootstrap confidence intervals (n={args.bootstrap})')
    ci_results = bootstrap_ci(y_true, y_pred, y_prob, n_bootstraps=args.bootstrap)

    # Write enhanced report with confidence intervals
    out=Path('reports/baseline_metrics.md'); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,'w',encoding='utf-8') as f:
        f.write('# Baseline Metrics — AD vs HC (LOSO, SVM)\n\n')
        f.write(f"- N samples: {metrics['N']}\n")
        f.write(f"- Bootstrap iterations: {args.bootstrap}\n\n")

        f.write('## Performance Metrics with 95% Confidence Intervals\n\n')
        f.write(f"- **F1 Score**: {metrics['F1']:.3f} (95% CI: [{ci_results['F1_CI'][0]:.3f}, {ci_results['F1_CI'][1]:.3f}])\n")
        f.write(f"- **MCC**: {metrics['MCC']:.3f} (95% CI: [{ci_results['MCC_CI'][0]:.3f}, {ci_results['MCC_CI'][1]:.3f}])\n")
        f.write(f"- **AUC**: {metrics['AUC']:.3f} (95% CI: [{ci_results['AUC_CI'][0]:.3f}, {ci_results['AUC_CI'][1]:.3f}])\n\n")

        f.write('## Confusion Matrices by Fold (0=HC,1=AD)\n')
        for i, cm in enumerate(cms,1):
            tn,fp,fn,tp = cm.ravel()
            f.write(f"- Fold {i}: TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")

        f.write('\n## Notes\n')
        f.write('- Confidence intervals computed using bootstrap resampling (2000 iterations)\n')
        f.write('- Each fold represents leave-one-subject-out cross-validation\n')
        f.write('- Poor performance due to minimal sample size (4 subjects)\n')

    print('[OK] Wrote reports/baseline_metrics.md with 95% confidence intervals')

if __name__=='__main__':
    main()
