#!/usr/bin/env python
"""
Spectrum feature extraction for EEG data
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import mne
from scipy import signal
from scipy.integrate import simpson
import yaml
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_band_power(psd: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]) -> float:
    """
    Compute power in specific frequency band

    Parameters
    ----------
    psd : array
        Power spectral density
    freqs : array
        Frequency values
    band : tuple
        Frequency band (low, high)

    Returns
    -------
    power : float
        Band power
    """
    band_freqs = np.logical_and(freqs >= band[0], freqs <= band[1])
    band_power = simpson(psd[band_freqs], x=freqs[band_freqs])
    return band_power

def extract_spectrum_features(raw: mne.io.Raw, config: Dict) -> Dict:
    """
    Extract spectrum features from EEG data

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG data
    config : dict
        Configuration with frequency bands

    Returns
    -------
    features : dict
        Dictionary of spectrum features
    """
    features = {}

    # Get EEG channels
    picks = mne.pick_types(raw.info, meg=False, eeg=True)

    # Compute PSD using Welch's method
    psd = raw.compute_psd(
        method='welch',
        picks=picks,
        fmin=0.5, fmax=45,
        n_fft=2048,
        n_overlap=1024,
        verbose=False
    )
    psds, freqs = psd.get_data(return_freqs=True)

    # Get frequency bands from config
    bands = config.get('bands', {
        'delta': [1, 4],
        'theta': [4, 8],
        'alpha': [8, 13],
        'beta': [13, 30]
    })

    # For each channel
    ch_names = [raw.ch_names[i] for i in picks]

    for ch_idx, ch_name in enumerate(ch_names):
        psd = psds[ch_idx]

        # Absolute power for each band
        for band_name, band_range in bands.items():
            power = compute_band_power(psd, freqs, band_range)
            features[f'{ch_name}_power_{band_name}'] = power

        # Total power
        total_power = simpson(psd, x=freqs)
        features[f'{ch_name}_power_total'] = total_power

        # Relative power for each band
        for band_name, band_range in bands.items():
            power = compute_band_power(psd, freqs, band_range)
            rel_power = power / total_power if total_power > 0 else 0
            features[f'{ch_name}_relpower_{band_name}'] = rel_power

        # Peak frequency in alpha band
        alpha_range = bands.get('alpha', [8, 13])
        alpha_freqs = np.logical_and(freqs >= alpha_range[0], freqs <= alpha_range[1])
        if np.any(alpha_freqs):
            alpha_psd = psd[alpha_freqs]
            alpha_freq = freqs[alpha_freqs]
            peak_idx = np.argmax(alpha_psd)
            features[f'{ch_name}_peak_alpha_freq'] = alpha_freq[peak_idx]

        # Spectral entropy
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-15))
        features[f'{ch_name}_spectral_entropy'] = spectral_entropy

    # Global features (average across channels)
    avg_psds = np.mean(psds, axis=0)

    for band_name, band_range in bands.items():
        power = compute_band_power(avg_psds, freqs, band_range)
        features[f'global_power_{band_name}'] = power

    total_power = simpson(avg_psds, x=freqs)
    features['global_power_total'] = total_power

    for band_name, band_range in bands.items():
        power = compute_band_power(avg_psds, freqs, band_range)
        rel_power = power / total_power if total_power > 0 else 0
        features[f'global_relpower_{band_name}'] = rel_power

    # Power ratios
    if 'theta' in bands and 'beta' in bands:
        theta_power = compute_band_power(avg_psds, freqs, bands['theta'])
        beta_power = compute_band_power(avg_psds, freqs, bands['beta'])
        features['global_theta_beta_ratio'] = theta_power / (beta_power + 1e-15)

    if 'theta' in bands and 'alpha' in bands:
        theta_power = compute_band_power(avg_psds, freqs, bands['theta'])
        alpha_power = compute_band_power(avg_psds, freqs, bands['alpha'])
        features['global_theta_alpha_ratio'] = theta_power / (alpha_power + 1e-15)

    return features

def main():
    parser = argparse.ArgumentParser(description='Extract spectrum features from EEG data')
    parser.add_argument('--input_dir', type=str, default='data/derivatives/clean',
                       help='Directory with preprocessed EEG files')
    parser.add_argument('--config', type=str, default='configs/bands.yaml',
                       help='Configuration file with frequency bands')
    parser.add_argument('--output_file', type=str, default='data/derivatives/features_spectrum.parquet',
                       help='Output file for features')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

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
            features = extract_spectrum_features(raw, config)
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
    print(f"  - Power features: {sum('power' in c for c in df.columns)}")
    print(f"  - Relative power: {sum('relpower' in c for c in df.columns)}")
    print(f"  - Peak frequency: {sum('peak' in c for c in df.columns)}")
    print(f"  - Spectral entropy: {sum('entropy' in c for c in df.columns)}")
    print(f"  - Power ratios: {sum('ratio' in c for c in df.columns)}")

if __name__ == '__main__':
    main()
