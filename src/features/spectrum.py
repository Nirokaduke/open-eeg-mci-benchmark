"""
Spectral feature extraction module for EEG-MCI-Bench.

Implements power spectral density (PSD) computation and spectral features
including relative power in frequency bands and peak frequency detection.

Functions:
    compute_psd: Compute power spectral density using Welch method
    compute_relative_power: Calculate relative power in frequency bands
    compute_peak_frequency: Find peak frequency in specified range
    extract_spectral_features: Extract all spectral features into DataFrame
"""

import argparse
import logging
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import yaml
import mne
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_psd(
    raw: mne.io.BaseRaw,
    fmin: float = 1.0,
    fmax: float = 40.0,
    method: str = 'welch',
    n_fft: Optional[int] = None,
    n_overlap: Optional[int] = None,
    n_per_seg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object containing EEG data
    fmin : float, default=1.0
        Minimum frequency for PSD computation (Hz)
    fmax : float, default=40.0
        Maximum frequency for PSD computation (Hz)
    method : str, default='welch'
        Method for PSD computation ('welch' or 'multitaper')
    n_fft : int, optional
        Length of FFT window. If None, uses 2 * sfreq
    n_overlap : int, optional
        Number of overlapping samples. If None, uses n_fft // 2
    n_per_seg : int, optional
        Length of each Welch segment. If None, uses 2 * sfreq

    Returns
    -------
    freqs : np.ndarray
        Array of frequencies (Hz)
    psd : np.ndarray
        Power spectral density array (channels x frequencies)

    Raises
    ------
    ValueError
        If method is not supported or frequency range is invalid

    Examples
    --------
    >>> freqs, psd = compute_psd(raw, fmin=1, fmax=40)
    >>> peak_freq = freqs[np.argmax(psd[0])]
    """
    if method not in ['welch', 'multitaper']:
        raise ValueError(f"Method '{method}' not supported. Use 'welch' or 'multitaper'.")

    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")

    logger.info(f"Computing PSD using {method} method for {fmin}-{fmax} Hz")

    try:
        # Build keyword arguments, only including non-None values
        psd_kwargs = {
            'method': method,
            'fmin': fmin,
            'fmax': fmax,
            'verbose': False
        }

        if n_fft is not None:
            psd_kwargs['n_fft'] = n_fft
        if n_overlap is not None:
            psd_kwargs['n_overlap'] = n_overlap
        if n_per_seg is not None:
            psd_kwargs['n_per_seg'] = n_per_seg

        # Use MNE's compute_psd method
        spectrum = raw.compute_psd(**psd_kwargs)

        # Extract frequencies and PSD values
        freqs = spectrum.freqs
        psd = spectrum.get_data()

        logger.info(f"PSD computed: {psd.shape[0]} channels, {len(freqs)} frequency bins")
        return freqs, psd

    except Exception as e:
        logger.error(f"Error computing PSD: {str(e)}")
        raise


def compute_relative_power(
    raw: mne.io.BaseRaw,
    bands: Dict[str, Tuple[float, float]],
    method: str = 'welch'
) -> Dict[str, np.ndarray]:
    """
    Calculate relative power in specified frequency bands.

    Relative power is the power in each band divided by the total power
    across all bands, providing a normalized measure of band activity.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object containing EEG data
    bands : dict
        Dictionary mapping band names to (fmin, fmax) tuples
        Example: {'alpha': (8, 13), 'beta': (13, 30)}
    method : str, default='welch'
        Method for PSD computation

    Returns
    -------
    rel_power : dict
        Dictionary mapping band names to relative power arrays (n_channels,)

    Examples
    --------
    >>> bands = {'alpha': (8, 13), 'beta': (13, 30)}
    >>> rel_power = compute_relative_power(raw, bands)
    >>> alpha_power = rel_power['alpha']
    """
    logger.info(f"Computing relative power for {len(bands)} frequency bands")

    # Get overall frequency range
    all_freqs = [f for band in bands.values() for f in band]
    fmin_all = min(all_freqs)
    fmax_all = max(all_freqs)

    # Compute PSD across full range
    freqs, psd = compute_psd(raw, fmin=fmin_all, fmax=fmax_all, method=method)

    # Calculate power in each band
    band_powers = {}
    for band_name, (fmin, fmax) in bands.items():
        # Find frequency indices for this band
        freq_mask = (freqs >= fmin) & (freqs <= fmax)

        # Integrate power in this band (using trapezoidal rule)
        band_power = np.trapezoid(psd[:, freq_mask], freqs[freq_mask], axis=1)
        band_powers[band_name] = band_power

    # Calculate total power across all bands
    total_power = sum(band_powers.values())

    # Compute relative power for each band
    rel_power = {}
    for band_name, power in band_powers.items():
        # Add small epsilon to avoid division by zero
        rel_power[band_name] = power / (total_power + 1e-10)
        logger.debug(f"{band_name} relative power: mean={np.mean(rel_power[band_name]):.4f}")

    logger.info("Relative power computation completed")
    return rel_power


def compute_peak_frequency(
    raw: mne.io.BaseRaw,
    fmin: float = 8.0,
    fmax: float = 13.0,
    method: str = 'welch'
) -> np.ndarray:
    """
    Find peak frequency in specified frequency range for each channel.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object containing EEG data
    fmin : float, default=8.0
        Minimum frequency of search range (Hz)
    fmax : float, default=13.0
        Maximum frequency of search range (Hz)
    method : str, default='welch'
        Method for PSD computation

    Returns
    -------
    peak_freqs : np.ndarray
        Array of peak frequencies for each channel (n_channels,)

    Examples
    --------
    >>> peak_freqs = compute_peak_frequency(raw, fmin=8, fmax=13)
    >>> alpha_peak = np.mean(peak_freqs)  # Average alpha peak across channels
    """
    logger.info(f"Computing peak frequencies in {fmin}-{fmax} Hz range")

    # Compute PSD in specified range
    freqs, psd = compute_psd(raw, fmin=fmin, fmax=fmax, method=method)

    # Find peak frequency for each channel
    peak_indices = np.argmax(psd, axis=1)
    peak_freqs = freqs[peak_indices]

    logger.info(f"Peak frequency: mean={np.mean(peak_freqs):.2f} Hz, "
                f"std={np.std(peak_freqs):.2f} Hz")

    return peak_freqs


def extract_spectral_features(
    raw: mne.io.BaseRaw,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    method: str = 'welch',
    compute_peak_bands: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract comprehensive spectral features into a DataFrame.

    Computes absolute power, relative power, and peak frequencies for
    specified frequency bands, organized by channel.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object containing EEG data
    bands : dict, optional
        Dictionary mapping band names to (fmin, fmax) tuples.
        If None, uses standard EEG bands:
        {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'gamma': (30, 45)}
    method : str, default='welch'
        Method for PSD computation
    compute_peak_bands : list, optional
        List of band names for which to compute peak frequencies.
        If None, computes for all bands.

    Returns
    -------
    features : pd.DataFrame
        DataFrame with spectral features, one row per channel.
        Columns include:
        - {band}_power: Absolute power in band
        - {band}_rel_power: Relative power in band
        - peak_freq_{band}: Peak frequency in band (if requested)

    Examples
    --------
    >>> bands = {'alpha': (8, 13), 'beta': (13, 30)}
    >>> features = extract_spectral_features(raw, bands=bands)
    >>> print(features[['alpha_power', 'alpha_rel_power']].head())
    """
    logger.info("Extracting comprehensive spectral features")

    # Use standard bands if none provided
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

    # Get channel names
    ch_names = raw.ch_names
    n_channels = len(ch_names)

    # Initialize feature dictionary
    feature_dict = {'channel': ch_names}

    # 1. Compute absolute power for each band
    logger.info("Computing absolute power...")
    all_freqs = [f for band in bands.values() for f in band]
    fmin_all = min(all_freqs)
    fmax_all = max(all_freqs)
    freqs, psd = compute_psd(raw, fmin=fmin_all, fmax=fmax_all, method=method)

    for band_name, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.trapezoid(psd[:, freq_mask], freqs[freq_mask], axis=1)
        feature_dict[f'{band_name}_power'] = band_power

    # 2. Compute relative power
    logger.info("Computing relative power...")
    rel_power = compute_relative_power(raw, bands, method=method)
    for band_name, power in rel_power.items():
        feature_dict[f'{band_name}_rel_power'] = power

    # 3. Compute peak frequencies for specified bands
    if compute_peak_bands is None:
        compute_peak_bands = list(bands.keys())

    logger.info(f"Computing peak frequencies for {len(compute_peak_bands)} bands...")
    for band_name in compute_peak_bands:
        if band_name in bands:
            fmin, fmax = bands[band_name]
            peak_freqs = compute_peak_frequency(raw, fmin=fmin, fmax=fmax, method=method)
            feature_dict[f'peak_freq_{band_name}'] = peak_freqs
        else:
            logger.warning(f"Band '{band_name}' not found in bands dictionary, skipping")

    # Create DataFrame
    features = pd.DataFrame(feature_dict)

    logger.info(f"Extracted {len(features.columns)-1} spectral features for {n_channels} channels")
    return features


def main(cfg: str = 'configs/bands.yaml') -> None:
    """
    Main function for batch spectral feature extraction.

    Loads configuration, processes EEG data, and saves features to derivatives.

    Parameters
    ----------
    cfg : str, default='configs/bands.yaml'
        Path to configuration file containing frequency bands
    """
    logger.info(f"Loading configuration from {cfg}")

    # Load band configuration
    with open(cfg, 'r') as f:
        config = yaml.safe_load(f)

    bands = config.get('bands', {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    })

    logger.info(f'Computing spectrum features for bands: {list(bands.keys())}')

    # TODO: Load BIDS data and iterate through subjects/sessions
    # TODO: Extract features and save to data/derivatives/features/
    logger.warning("Batch processing not yet implemented. Use individual functions.")

    print(f'Configured bands: {bands}')
    print('To use this module:')
    print('  from src.features.spectrum import extract_spectral_features')
    print('  features = extract_spectral_features(raw, bands=bands)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Extract spectral features from EEG data'
    )
    ap.add_argument(
        '--config',
        default='configs/bands.yaml',
        help='Path to configuration file with frequency bands'
    )
    args = ap.parse_args()
    main(args.config)
