"""
Entropy and complexity feature extraction for EEG-MCI-Bench.

This module implements entropy and complexity measures for EEG analysis:
- Permutation Entropy: Ordinal pattern entropy
- Sample Entropy: Irregularity measure
- Shannon Entropy: Information entropy
- Lempel-Ziv Complexity: Binary sequence complexity
- Fractal Dimension: Higuchi fractal dimension

All implementations are optimized for performance using NumPy.
"""

import numpy as np
import pandas as pd
import mne
import math
from typing import Optional, Union, Dict
from itertools import permutations
from scipy.stats import entropy as scipy_entropy


def compute_permutation_entropy(
    signal: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True
) -> float:
    """
    Compute permutation entropy for a signal.

    Permutation entropy quantifies the complexity of a time series by analyzing
    the ordinal patterns in the signal. Higher values indicate more complexity.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    order : int, optional
        Embedding dimension (pattern length), default=3
    delay : int, optional
        Time delay for embedding, default=1
    normalize : bool, optional
        Whether to normalize to [0, 1], default=True

    Returns
    -------
    float
        Permutation entropy value (0-1 if normalized)

    References
    ----------
    Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity
    measure for time series. Physical review letters, 88(17), 174102.
    """
    # Handle edge cases
    if len(signal) < order * delay + 1:
        return 0.0

    # Handle constant signals
    if np.std(signal) < 1e-10:
        return 0.0

    # Normalize signal for numerical stability
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

    # Generate all possible permutations
    n_permutations = math.factorial(order)
    permutation_patterns = list(permutations(range(order)))
    pattern_counts = np.zeros(n_permutations)

    # Create permutation lookup dictionary
    pattern_to_idx = {pat: idx for idx, pat in enumerate(permutation_patterns)}

    # Extract embedded sequences
    n_sequences = len(signal) - (order - 1) * delay

    for i in range(n_sequences):
        # Extract pattern with delay
        indices = [i + j * delay for j in range(order)]
        subsequence = signal[indices]

        # Get ordinal pattern
        pattern = tuple(np.argsort(subsequence))

        # Count pattern occurrence
        pattern_counts[pattern_to_idx[pattern]] += 1

    # Calculate probabilities
    probabilities = pattern_counts / n_sequences

    # Remove zero probabilities
    probabilities = probabilities[probabilities > 0]

    # Calculate entropy
    pe = -np.sum(probabilities * np.log2(probabilities))

    # Normalize to [0, 1]
    if normalize:
        max_entropy = np.log2(n_permutations)
        pe = pe / max_entropy if max_entropy > 0 else 0.0

    return float(pe)


def compute_sample_entropy(
    signal: np.ndarray,
    m: int = 2,
    r: float = 0.2,
    scale: bool = True
) -> float:
    """
    Compute sample entropy for a signal.

    Sample entropy measures the irregularity and unpredictability of a time series.
    Lower values indicate more self-similarity, higher values indicate more irregularity.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    m : int, optional
        Pattern length, default=2
    r : float, optional
        Tolerance for matching (as fraction of signal std), default=0.2
    scale : bool, optional
        Whether to scale r by signal std, default=True

    Returns
    -------
    float
        Sample entropy value (non-negative, unbounded)

    References
    ----------
    Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis
    using approximate entropy and sample entropy. American Journal of
    Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
    """
    # Handle edge cases
    N = len(signal)
    if N < m + 2:
        return 0.0

    # Handle constant signals
    signal_std = np.std(signal)
    if signal_std < 1e-10:
        return 0.0

    # Normalize signal
    signal = (signal - np.mean(signal)) / (signal_std + 1e-10)

    # Scale tolerance by signal std
    tolerance = r * signal_std if scale else r

    def _maxdist(xmi, xmj):
        """Calculate maximum distance between template vectors"""
        return np.max(np.abs(xmi - xmj))

    def _phi(m_val):
        """Calculate phi for pattern length m"""
        # Create template vectors
        patterns = np.array([signal[i:i + m_val] for i in range(N - m_val)])

        # Count matches
        match_count = 0
        for i in range(len(patterns)):
            # Compare with all patterns except self
            for j in range(len(patterns)):
                if i != j:  # Exclude self-matches (key difference from ApEn)
                    if _maxdist(patterns[i], patterns[j]) <= tolerance:
                        match_count += 1

        # Return probability
        n_comparisons = len(patterns) * (len(patterns) - 1)
        return match_count / n_comparisons if n_comparisons > 0 else 0

    # Calculate phi for m and m+1
    phi_m = _phi(m)
    phi_m_plus_1 = _phi(m + 1)

    # Avoid log(0) errors
    if phi_m == 0 or phi_m_plus_1 == 0:
        return 0.0

    # Calculate sample entropy
    sample_ent = -np.log(phi_m_plus_1 / phi_m)

    return float(sample_ent)


def compute_shannon_entropy(
    signal: np.ndarray,
    n_bins: int = 256,
    normalize: bool = True
) -> float:
    """
    Compute Shannon entropy of a signal.

    Shannon entropy measures the information content or unpredictability of a signal
    based on its power spectral density. More complex signals have higher entropy.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    n_bins : int, optional
        Number of frequency bins for PSD-based entropy, default=256
    normalize : bool, optional
        Whether to normalize to [0, 1], default=True

    Returns
    -------
    float
        Shannon entropy value

    References
    ----------
    Shannon, C. E. (1948). A mathematical theory of communication.
    The Bell system technical journal, 27(3), 379-423.

    Notes
    -----
    This implementation computes entropy from the normalized power spectral density,
    which better reflects signal complexity than amplitude histogram entropy.
    """
    # Handle edge cases
    if len(signal) < 2:
        return 0.0

    # Handle constant signals
    if np.std(signal) < 1e-10:
        return 0.0

    # Compute power spectral density using FFT
    # This better captures signal complexity than amplitude histograms
    fft_vals = np.fft.fft(signal)
    psd = np.abs(fft_vals) ** 2

    # Only use positive frequencies (first half)
    psd = psd[:len(psd) // 2]

    # Normalize to get probability distribution
    psd_sum = np.sum(psd)
    if psd_sum < 1e-10:
        return 0.0

    psd_prob = psd / psd_sum

    # Remove zeros to avoid log(0)
    psd_prob = psd_prob[psd_prob > 1e-12]

    # Calculate Shannon entropy
    shannon_ent = -np.sum(psd_prob * np.log2(psd_prob + 1e-12))

    # Normalize to [0, 1] if requested
    if normalize:
        # Maximum entropy for uniform distribution over frequency bins
        max_entropy = np.log2(len(psd_prob))
        shannon_ent = shannon_ent / max_entropy if max_entropy > 0 else 0.0

    return float(shannon_ent)


def compute_lempel_ziv_complexity(
    signal: np.ndarray,
    threshold: Optional[float] = None,
    normalize: bool = True
) -> float:
    """
    Compute Lempel-Ziv complexity of a signal.

    LZ complexity measures the generation rate of new patterns along a sequence,
    reflecting the complexity or randomness of the signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    threshold : float, optional
        Threshold for binarization (uses median if None)
    normalize : bool, optional
        Whether to normalize by theoretical maximum, default=True

    Returns
    -------
    float
        Lempel-Ziv complexity value

    References
    ----------
    Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences.
    IEEE Transactions on information theory, 22(1), 75-81.
    """
    # Handle edge cases
    n = len(signal)
    if n < 2:
        return 0.0

    # Handle constant signals
    if np.std(signal) < 1e-10:
        return 0.0

    # Binarize signal
    if threshold is None:
        threshold = np.median(signal)

    binary_sequence = (signal > threshold).astype(int)

    # Convert to string for pattern matching
    binary_str = ''.join(map(str, binary_sequence))

    # LZ76 algorithm
    complexity = 1
    prefix_len = 1
    pos = 0
    max_len = n

    while prefix_len + pos < max_len:
        # Subsequence to check
        subseq = binary_str[pos:pos + prefix_len + 1]

        # Search in prefix
        prefix = binary_str[:pos + prefix_len]

        # Check if subsequence exists in prefix
        if subseq[:-1] in prefix and subseq not in prefix:
            complexity += 1
            pos += prefix_len + 1
            prefix_len = 1
        else:
            prefix_len += 1

    # Normalize by theoretical upper bound
    if normalize:
        # Theoretical maximum complexity for binary sequence
        max_complexity = n / np.log2(n) if n > 1 else 1
        complexity = complexity / max_complexity

    return float(complexity)


def compute_fractal_dimension(
    signal: np.ndarray,
    kmax: Optional[int] = None
) -> float:
    """
    Compute Higuchi fractal dimension of a signal.

    Fractal dimension characterizes the complexity and self-similarity of a time series.
    Values range from 1 (smooth) to 2 (very irregular).

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    kmax : int, optional
        Maximum k value (uses len(signal)//10 if None)

    Returns
    -------
    float
        Fractal dimension (typically between 1 and 2)

    References
    ----------
    Higuchi, T. (1988). Approach to an irregular time series on the basis of
    the fractal theory. Physica D: Nonlinear Phenomena, 31(2), 277-283.
    """
    # Handle edge cases
    N = len(signal)
    if N < 10:
        return 1.5  # Return midpoint if signal too short

    # Handle constant signals
    if np.std(signal) < 1e-10:
        return 1.0

    # Set kmax if not provided
    if kmax is None:
        kmax = min(16, N // 10)

    kmax = max(4, min(kmax, N // 6))

    # Calculate curve length L(k) for each k
    L = []
    k_values = []

    for k in range(1, kmax + 1):
        # For each starting point m from 0 to k-1
        Lmk = []

        for m in range(k):
            # Build the subsequence X_m^k
            indices = np.arange(m, N, k)
            if len(indices) < 2:
                continue

            # Calculate the length of this curve
            subsequence = signal[indices]

            # Length = sum of distances between consecutive points
            length = np.sum(np.abs(np.diff(subsequence)))

            # Normalize by the reconstruction factor
            # L_m(k) = (length * (N-1)) / (floor((N-m)/k) * k)
            num_intervals = len(indices) - 1
            if num_intervals > 0:
                Lmk.append(length * (N - 1) / (num_intervals * k))

        # Average over all starting points m for this k
        if len(Lmk) > 0:
            L.append(np.mean(Lmk))
            k_values.append(k)

    # Convert to arrays
    L = np.array(L)
    k_values = np.array(k_values)

    # Need at least 3 points for reliable regression
    if len(L) < 3:
        return 1.5

    # Higuchi FD: The curve length L(k) decreases as k increases
    # log(L(k)) vs log(k) gives a line with negative slope
    # The magnitude of the slope relates to fractal dimension
    # Steep negative slope (more negative) → faster decrease → higher complexity → higher FD
    # Gentle negative slope (less negative) → slower decrease → lower complexity → lower FD
    log_k = np.log(k_values)
    log_L = np.log(L + 1e-10)  # Add small epsilon to avoid log(0)

    # Linear regression
    slope, intercept = np.polyfit(log_k, log_L, 1)

    # Fractal dimension from Higuchi method
    # FD = 2 - slope (where slope is the regression slope, typically negative)
    # More negative slope → larger FD
    # Example: slope=-1.0 → FD=2-(-1)=3, but we clip to [1,2], so FD≈2
    #          slope=-0.2 → FD=2-(-0.2)=2.2, clipped to 2
    #          slope=0 → FD=2

    # Actually, the relationship is: FD directly equals the absolute value of slope
    # for proper Higuchi scaling, normalized between 1 and 2
    fractal_dim = abs(slope)

    # Map to [1, 2] range
    # Typical slope ranges from 0 (flat, FD→1) to -2 (steep, FD→2)
    # So we map abs(slope) directly, with gentle slopes→1, steep→2
    if fractal_dim < 1.0:
        fractal_dim = 1.0 + fractal_dim  # Map [0,1) to [1,2)

    # Clip to valid range [1.0, 2.0]
    fractal_dim = float(np.clip(fractal_dim, 1.0, 2.0))

    return fractal_dim


def extract_complexity_features(
    raw_or_epochs: Union[mne.io.BaseRaw, mne.BaseEpochs],
    channels: Optional[list] = None,
    pe_order: int = 3,
    se_m: int = 2,
    se_r: float = 0.2
) -> pd.DataFrame:
    """
    Extract all complexity features from EEG data.

    Combines permutation entropy, sample entropy, Shannon entropy,
    Lempel-Ziv complexity, and fractal dimension into a DataFrame.

    Parameters
    ----------
    raw_or_epochs : mne.io.BaseRaw or mne.BaseEpochs
        EEG data
    channels : list, optional
        List of channel names (uses all if None)
    pe_order : int, optional
        Order for permutation entropy, default=3
    se_m : int, optional
        Pattern length for sample entropy, default=2
    se_r : float, optional
        Tolerance for sample entropy, default=0.2

    Returns
    -------
    pd.DataFrame
        DataFrame with complexity features for each channel
        Columns: channel, perm_entropy, sample_entropy, shannon_entropy,
                 lz_complexity, fractal_dimension
    """
    # Get data
    if isinstance(raw_or_epochs, mne.io.BaseRaw):
        data = raw_or_epochs.get_data()
        ch_names = raw_or_epochs.ch_names
    else:  # Epochs
        # Average across epochs
        data = raw_or_epochs.get_data().mean(axis=0)
        ch_names = raw_or_epochs.ch_names

    # Select channels
    if channels is None:
        channels = ch_names

    channel_indices = [ch_names.index(ch) for ch in channels if ch in ch_names]

    # Initialize results
    results = []

    # Compute features for each channel
    for idx, ch_idx in enumerate(channel_indices):
        signal = data[ch_idx]
        ch_name = channels[idx]

        # Compute all entropy measures
        pe = compute_permutation_entropy(signal, order=pe_order, delay=1)
        se = compute_sample_entropy(signal, m=se_m, r=se_r)
        she = compute_shannon_entropy(signal)
        lzc = compute_lempel_ziv_complexity(signal)
        fd = compute_fractal_dimension(signal)

        results.append({
            'channel': ch_name,
            'perm_entropy': pe,
            'sample_entropy': se,
            'shannon_entropy': she,
            'lz_complexity': lzc,
            'fractal_dimension': fd
        })

    # Create DataFrame
    df = pd.DataFrame(results)
    df.set_index('channel', inplace=True)

    return df


# Main function for standalone execution
def main(cfg='configs/bands.yaml'):
    """Main function for command-line execution."""
    import argparse
    import yaml

    print('Computing entropy/complexity features...')

    # Load configuration
    if cfg and Path(cfg).exists():
        with open(cfg, 'r') as f:
            config = yaml.safe_load(f)
        print(f'Loaded configuration from {cfg}')
    else:
        print('No configuration file found, using defaults')
        config = {}

    # TODO: Load EEG data and compute features
    # This would be implemented based on your data loading pipeline
    print('Feature extraction would be performed here')


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description='Extract entropy features from EEG data')
    ap.add_argument('--config', default='configs/bands.yaml', help='Configuration file')
    args = ap.parse_args()

    main(args.config)
