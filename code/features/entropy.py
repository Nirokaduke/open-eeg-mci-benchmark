#!/usr/bin/env python
"""
Entropy and complexity feature extraction for EEG data
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import mne
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

def permutation_entropy(signal: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Calculate permutation entropy of a time series

    Parameters
    ----------
    signal : array
        Input signal
    order : int
        Embedding dimension
    delay : int
        Time delay

    Returns
    -------
    pe : float
        Permutation entropy
    """
    n = len(signal)
    if n < order:
        return np.nan

    # Create embedding matrix
    embed_matrix = np.zeros((n - (order - 1) * delay, order))
    for i in range(order):
        embed_matrix[:, i] = signal[i * delay:n - (order - 1 - i) * delay]

    # Get permutation patterns
    permutations = np.argsort(embed_matrix, axis=1)

    # Convert to unique identifiers
    hash_vals = permutations @ (np.arange(order) ** np.arange(order)[:, None]).T[0]

    # Count occurrences
    unique, counts = np.unique(hash_vals, return_counts=True)
    probs = counts / len(hash_vals)

    # Calculate entropy
    pe = -np.sum(probs * np.log2(probs + 1e-15))

    # Normalize
    import math
    pe_norm = pe / np.log2(math.factorial(order))

    return pe_norm

def sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Calculate sample entropy

    Parameters
    ----------
    signal : array
        Input signal
    m : int
        Pattern length
    r : float
        Tolerance for matches (proportion of std)

    Returns
    -------
    sampen : float
        Sample entropy
    """
    N = len(signal)
    if N < m + 1:
        return np.nan

    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-15)
    r = r * np.std(signal)

    def _maxdist(xi, xj, m):
        return max([abs(float(xi[k]) - float(xj[k])) for k in range(m)])

    def _phi(m):
        patterns = [signal[i:i + m] for i in range(N - m + 1)]
        C = 0
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                if _maxdist(patterns[i], patterns[j], m) <= r:
                    C += 1
        return C

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0:
        return np.nan

    return -np.log(phi_m1 / phi_m) if phi_m1 > 0 else np.inf

def lempel_ziv_complexity(signal: np.ndarray) -> float:
    """
    Calculate Lempel-Ziv complexity

    Parameters
    ----------
    signal : array
        Input signal

    Returns
    -------
    lzc : float
        Normalized LZ complexity
    """
    # Binarize signal
    threshold = np.median(signal)
    binary = (signal > threshold).astype(int)

    n = len(binary)
    s = ''.join(map(str, binary))

    i, k, l = 0, 1, 1
    c = 1
    k_max = 1

    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k

            i += 1

            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1

    # Normalize
    b = n / np.log2(n)
    lzc_norm = c / b

    return lzc_norm

def higuchi_fractal_dimension(signal: np.ndarray, k_max: int = 10) -> float:
    """
    Calculate Higuchi fractal dimension

    Parameters
    ----------
    signal : array
        Input signal
    k_max : int
        Maximum k value

    Returns
    -------
    hfd : float
        Higuchi fractal dimension
    """
    N = len(signal)
    L = []

    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            Lmk = 0
            for i in range(1, int((N - m) / k)):
                Lmk += abs(signal[m + i * k] - signal[m + (i - 1) * k])
            Lmk = Lmk * (N - 1) / (k * int((N - m) / k))
            Lk.append(Lmk)
        L.append(np.mean(Lk))

    # Linear fit in log-log space
    x = np.log(1 / np.arange(1, k_max + 1))
    y = np.log(L)

    # Remove invalid values
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan

    x, y = x[mask], y[mask]

    # Fit line
    coeffs = np.polyfit(x, y, 1)
    hfd = coeffs[0]

    return hfd

def extract_entropy_features(raw: mne.io.Raw) -> Dict:
    """
    Extract entropy and complexity features from EEG data

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG data

    Returns
    -------
    features : dict
        Dictionary of entropy features
    """
    features = {}

    # Get EEG channels
    picks = mne.pick_types(raw.info, meg=False, eeg=True)
    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[i] for i in picks]

    for ch_idx, ch_name in enumerate(ch_names):
        signal = data[ch_idx]

        # Shannon entropy
        hist, _ = np.histogram(signal, bins=50)
        hist = hist / np.sum(hist)
        shannon = -np.sum(hist * np.log2(hist + 1e-15))
        features[f'{ch_name}_shannon_entropy'] = shannon

        # Permutation entropy
        pe = permutation_entropy(signal, order=3, delay=1)
        features[f'{ch_name}_permutation_entropy'] = pe

        # Sample entropy (subsample for speed)
        subsample = signal[::10]  # Downsample for computational efficiency
        sampen = sample_entropy(subsample, m=2, r=0.2)
        features[f'{ch_name}_sample_entropy'] = sampen

        # Lempel-Ziv complexity
        lzc = lempel_ziv_complexity(signal)
        features[f'{ch_name}_lzc'] = lzc

        # Higuchi fractal dimension
        hfd = higuchi_fractal_dimension(signal[::10], k_max=10)
        features[f'{ch_name}_higuchi_fd'] = hfd

    # Global features (average across channels)
    for measure in ['shannon_entropy', 'permutation_entropy', 'sample_entropy', 'lzc', 'higuchi_fd']:
        channel_values = [features[f'{ch}_{measure}'] for ch in ch_names
                         if not np.isnan(features.get(f'{ch}_{measure}', np.nan))]
        if channel_values:
            features[f'global_{measure}'] = np.mean(channel_values)
            features[f'global_{measure}_std'] = np.std(channel_values)

    return features

def main():
    parser = argparse.ArgumentParser(description='Extract entropy features from EEG data')
    parser.add_argument('--input_dir', type=str, default='data/derivatives/clean',
                       help='Directory with preprocessed EEG files')
    parser.add_argument('--output_file', type=str, default='data/derivatives/features_entropy.parquet',
                       help='Output file for features')

    args = parser.parse_args()

    # Find all preprocessed files
    input_dir = Path(args.input_dir)
    eeg_files = list(input_dir.glob('*_preprocessed_raw.fif'))

    if not eeg_files:
        print(f"No preprocessed EEG files found in {input_dir}")
        return

    print(f"Found {len(eeg_files)} preprocessed files")

    # Extract features for all subjects
    all_features = []

    for eeg_file in eeg_files:
        # Extract subject ID from filename
        subject_id = eeg_file.stem.split('_')[0]
        print(f"Processing {subject_id}...")

        try:
            # Load preprocessed data
            raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)

            # Extract features
            features = extract_entropy_features(raw)
            features['subject'] = subject_id

            all_features.append(features)
            print(f"  Extracted {len(features)-1} features")

        except Exception as e:
            print(f"  Error processing {subject_id}: {str(e)}")

    # Create DataFrame
    if all_features:
        df = pd.DataFrame(all_features)

        # Reorder columns to put subject first if it exists
        if 'subject' in df.columns:
            cols = df.columns.tolist()
            cols = ['subject'] + [c for c in cols if c != 'subject']
            df = df[cols]
    else:
        df = pd.DataFrame()

    # Save features
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"Feature extraction complete!")
    print(f"Processed: {len(all_features)} subjects")
    print(f"Features per subject: {len(df.columns)-1}")
    print(f"Output saved to: {output_path}")
    print(f"\nFeature categories:")
    print(f"  - Shannon entropy: {sum('shannon' in c for c in df.columns)}")
    print(f"  - Permutation entropy: {sum('permutation' in c for c in df.columns)}")
    print(f"  - Sample entropy: {sum('sample' in c for c in df.columns)}")
    print(f"  - LZ complexity: {sum('lzc' in c for c in df.columns)}")
    print(f"  - Fractal dimension: {sum('higuchi' in c for c in df.columns)}")

if __name__ == '__main__':
    main()
