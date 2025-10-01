#!/usr/bin/env python
"""
Merge all feature files into a single DataFrame
"""
import argparse
import pandas as pd
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Merge feature files')
    parser.add_argument('--input_dir', type=str, default='data/derivatives',
                       help='Directory with feature files')
    parser.add_argument('--output_file', type=str, default='data/derivatives/features.parquet',
                       help='Output file for merged features')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    feature_files = [
        'features_spectrum.parquet',
        'features_entropy.parquet',
        'features_connectivity.parquet'
    ]

    # Load and merge features
    dfs = []
    for feature_file in feature_files:
        file_path = input_dir / feature_file
        if file_path.exists():
            print(f"Loading {feature_file}...")
            df = pd.read_parquet(file_path)
            print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
            dfs.append(df)
        else:
            print(f"  {feature_file} not found, skipping")

    if not dfs:
        print("No feature files found!")
        return

    # Merge on subject column
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='subject', how='outer')

    # Save merged features
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"Feature merging complete!")
    print(f"Total subjects: {merged_df.shape[0]}")
    print(f"Total features: {merged_df.shape[1] - 1}")  # Minus subject column
    print(f"Output saved to: {output_path}")
    print(f"\nFeature breakdown:")

    # Count features by type
    spectrum_cols = [c for c in merged_df.columns if 'power' in c or 'relpower' in c or 'peak' in c or 'spectral' in c or 'ratio' in c]
    entropy_cols = [c for c in merged_df.columns if 'entropy' in c or 'lzc' in c or 'higuchi' in c or 'permutation' in c or 'sample' in c]
    conn_cols = [c for c in merged_df.columns if 'conn_' in c or 'graph_' in c]

    print(f"  - Spectrum features: {len(spectrum_cols)}")
    print(f"  - Entropy features: {len(entropy_cols)}")
    print(f"  - Connectivity features: {len(conn_cols)}")

if __name__ == '__main__':
    main()