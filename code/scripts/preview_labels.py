"""
Preview and validate labels from participants.tsv
Supports various label formats including A→AD, F→FTD, C→HC mappings
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.labels_ds004504 import load_participants_labels

def preview_labels_detailed(participants_path):
    """Enhanced preview with mapping validation"""

    print(f"\n{'='*60}")
    print(f"PARTICIPANTS.TSV LABEL PREVIEW")
    print(f"{'='*60}")
    print(f"File: {participants_path}")

    # Load using the utility function
    df, label_mapping = load_participants_labels(participants_path)

    print(f"\nTotal subjects: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Show raw data first
    print(f"\n[RAW DATA] Preview (first 7 rows):")
    print("-" * 40)
    print(df.head(7).to_string(index=False))

    # Show label distribution
    print(f"\n[LABEL DISTRIBUTION]:")
    print("-" * 40)
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {label}: {count} subjects ({percentage:.1f}%)")

    # Show label mappings used
    print(f"\n[LABEL MAPPINGS] Applied:")
    print("-" * 40)
    print("  A -> AD (Alzheimer's Disease)")
    print("  F -> FTD (Frontotemporal Dementia)")
    print("  C -> HC (Healthy Control)")

    # Show subject ID to label mapping
    print(f"\n[SUBJECT ID -> LABEL] Mapping:")
    print("-" * 40)
    for subj_id, label in sorted(label_mapping.items())[:10]:
        print(f"  {subj_id}: {label}")
    if len(label_mapping) > 10:
        print(f"  ... and {len(label_mapping)-10} more")

    # Binary classification info
    binary_df = df[df['label'].isin(['AD', 'HC'])]
    print(f"\n[BINARY CLASSIFICATION] AD vs HC:")
    print("-" * 40)
    print(f"  AD: {sum(binary_df['label'] == 'AD')} subjects")
    print(f"  HC: {sum(binary_df['label'] == 'HC')} subjects")
    print(f"  Total for binary: {len(binary_df)} subjects")

    excluded = len(df) - len(binary_df)
    if excluded > 0:
        excluded_labels = df[~df['label'].isin(['AD', 'HC'])]['label'].unique()
        print(f"  Excluded: {excluded} subjects ({', '.join(excluded_labels)})")

    # Check for metadata columns
    metadata_cols = [col for col in df.columns if col not in ['participant_id', 'label']]
    if metadata_cols:
        print(f"\n[ADDITIONAL METADATA]:")
        print("-" * 40)
        for col in metadata_cols:
            unique_vals = df[col].nunique()
            print(f"  {col}: {unique_vals} unique values")
            if unique_vals <= 5:
                print(f"    -> {df[col].value_counts().to_dict()}")

    print(f"\n{'='*60}")
    print("[OK] Label preview complete! Ready for LOSO-CV.")
    print(f"{'='*60}\n")

    return df, label_mapping

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Preview and validate participants.tsv labels')
    ap.add_argument('--participants', default='data/bids_raw/ds004504/participants.tsv',
                    help='Path to participants.tsv file')
    args=ap.parse_args()

    try:
        df, mapping = preview_labels_detailed(args.participants)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
