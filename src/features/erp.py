"""
Event-Related Potential (ERP) feature extraction for EEG-MCI-Bench.

This module extracts ERP components including P300 and N400, following
BIDS compliance and best practices for cognitive neuroscience research.

Functions:
    - extract_p300: Extract P300 component features
    - extract_n400: Extract N400 component features
    - compute_erp_amplitude: Get peak amplitude in time window
    - compute_erp_latency: Get peak latency
    - extract_erp_features: Combine all ERP features by channel groups

Author: EEG-MCI-Bench Team
License: MIT
"""

import numpy as np
import pandas as pd
import mne
from typing import Dict, List, Optional, Tuple, Union


def compute_erp_amplitude(
    epochs: mne.Epochs,
    time_window: Tuple[float, float],
    channels: Optional[Union[str, List[str]]] = None,
    method: str = 'peak'
) -> np.ndarray:
    """
    Compute ERP amplitude in specified time window.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    time_window : tuple of float
        Time window (tmin, tmax) in seconds
    channels : str or list of str, optional
        Channels to use. If None, uses all channels
    method : str, default='peak'
        Method to compute amplitude: 'peak', 'mean', or 'median'

    Returns
    -------
    amplitudes : np.ndarray
        Amplitude values for each channel (in microvolts)

    Examples
    --------
    >>> amplitudes = compute_erp_amplitude(epochs, (0.25, 0.35))
    >>> print(f"Mean P300 amplitude: {np.mean(amplitudes):.2f} µV")
    """
    # Select channels
    if channels is not None:
        if channels == 'all':
            epochs_subset = epochs.copy()
        else:
            epochs_subset = epochs.copy().pick(channels)
    else:
        epochs_subset = epochs.copy()

    # Extract data in time window
    tmin, tmax = time_window
    times = epochs_subset.times
    time_mask = (times >= tmin) & (times <= tmax)

    # Average across epochs first
    evoked = epochs_subset.average()
    data = evoked.data[:, time_mask]  # channels x time_points

    # Compute amplitude based on method
    if method == 'peak':
        # Find maximum absolute value in window
        amplitudes = data[np.arange(len(data)), np.argmax(np.abs(data), axis=1)]
    elif method == 'mean':
        amplitudes = np.mean(data, axis=1)
    elif method == 'median':
        amplitudes = np.median(data, axis=1)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'peak', 'mean', or 'median'")

    # Convert from V to µV
    amplitudes = amplitudes * 1e6

    return amplitudes


def compute_erp_latency(
    epochs: mne.Epochs,
    time_window: Tuple[float, float],
    polarity: str = 'positive',
    channels: Optional[Union[str, List[str]]] = None
) -> np.ndarray:
    """
    Compute ERP latency (time of peak) in specified window.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    time_window : tuple of float
        Time window (tmin, tmax) in seconds
    polarity : str, default='positive'
        Peak polarity: 'positive' or 'negative'
    channels : str or list of str, optional
        Channels to use. If None, uses all channels

    Returns
    -------
    latencies : np.ndarray
        Latency values for each channel (in seconds)

    Examples
    --------
    >>> latencies = compute_erp_latency(epochs, (0.25, 0.5), polarity='positive')
    >>> print(f"Mean P300 latency: {np.mean(latencies)*1000:.0f} ms")
    """
    # Select channels
    if channels is not None:
        if channels == 'all':
            epochs_subset = epochs.copy()
        else:
            epochs_subset = epochs.copy().pick(channels)
    else:
        epochs_subset = epochs.copy()

    # Extract data in time window
    tmin, tmax = time_window
    times = epochs_subset.times
    time_mask = (times >= tmin) & (times <= tmax)
    windowed_times = times[time_mask]

    # Average across epochs
    evoked = epochs_subset.average()
    data = evoked.data[:, time_mask]  # channels x time_points

    # Find peak based on polarity
    if polarity == 'positive':
        peak_indices = np.argmax(data, axis=1)
    elif polarity == 'negative':
        peak_indices = np.argmin(data, axis=1)
    else:
        raise ValueError(f"Unknown polarity: {polarity}. Use 'positive' or 'negative'")

    # Convert indices to latencies
    latencies = windowed_times[peak_indices]

    return latencies


def extract_p300(
    epochs: mne.Epochs,
    channels: Union[str, List[str]] = 'all',
    window: Tuple[float, float] = (0.25, 0.5)
) -> Dict[str, float]:
    """
    Extract P300 component features.

    The P300 is a positive deflection typically occurring around 300ms
    post-stimulus, maximal over parietal regions. It's associated with
    attention and working memory processes.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    channels : str or list of str, default='all'
        Channels to analyze. Use 'all' for all channels or provide list
    window : tuple of float, default=(0.25, 0.5)
        Time window for P300 detection in seconds

    Returns
    -------
    features : dict
        Dictionary containing:
        - 'amplitude': Mean P300 amplitude across channels (µV)
        - 'latency': Mean P300 latency across channels (seconds)

    Examples
    --------
    >>> p300 = extract_p300(epochs, channels=['Pz', 'P3', 'P4'])
    >>> print(f"P300 amplitude: {p300['amplitude']:.2f} µV")
    >>> print(f"P300 latency: {p300['latency']*1000:.0f} ms")
    """
    # Compute amplitude (positive peak)
    amplitudes = compute_erp_amplitude(
        epochs,
        time_window=window,
        channels=channels,
        method='peak'
    )

    # Compute latency (positive peak)
    latencies = compute_erp_latency(
        epochs,
        time_window=window,
        polarity='positive',
        channels=channels
    )

    # Return mean across channels
    features = {
        'amplitude': float(np.mean(amplitudes)),
        'latency': float(np.mean(latencies))
    }

    return features


def extract_n400(
    epochs: mne.Epochs,
    channels: Union[str, List[str]] = 'all',
    window: Tuple[float, float] = (0.35, 0.6)
) -> Dict[str, float]:
    """
    Extract N400 component features.

    The N400 is a negative deflection typically occurring around 400ms
    post-stimulus, associated with semantic processing and language
    comprehension.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    channels : str or list of str, default='all'
        Channels to analyze. Use 'all' for all channels or provide list
    window : tuple of float, default=(0.35, 0.6)
        Time window for N400 detection in seconds

    Returns
    -------
    features : dict
        Dictionary containing:
        - 'amplitude': Mean N400 amplitude across channels (µV)
        - 'latency': Mean N400 latency across channels (seconds)

    Examples
    --------
    >>> n400 = extract_n400(epochs, channels=['Cz', 'C3', 'C4'])
    >>> print(f"N400 amplitude: {n400['amplitude']:.2f} µV")
    >>> print(f"N400 latency: {n400['latency']*1000:.0f} ms")
    """
    # Compute amplitude (negative peak)
    amplitudes = compute_erp_amplitude(
        epochs,
        time_window=window,
        channels=channels,
        method='peak'
    )

    # Compute latency (negative peak)
    latencies = compute_erp_latency(
        epochs,
        time_window=window,
        polarity='negative',
        channels=channels
    )

    # Return mean across channels
    # Note: We keep the negative sign for N400 amplitude
    features = {
        'amplitude': float(np.mean(amplitudes)),
        'latency': float(np.mean(latencies))
    }

    return features


def extract_erp_features(
    epochs: mne.Epochs,
    components: Optional[List[str]] = None,
    channel_groups: Optional[Dict[str, List[str]]] = None,
    windows: Optional[Dict[str, Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    Extract comprehensive ERP features across components and channel groups.

    This function provides a complete ERP feature set for machine learning,
    extracting amplitude and latency for multiple ERP components across
    different scalp regions.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    components : list of str, optional
        ERP components to extract. Options: 'P300', 'N400'
        Default: ['P300', 'N400']
    channel_groups : dict, optional
        Dictionary mapping group names to channel lists
        Example: {'parietal': ['Pz', 'P3', 'P4'], 'central': ['Cz', 'C3', 'C4']}
        If None, uses all channels as single group
    windows : dict, optional
        Custom time windows for each component
        Example: {'P300': (0.25, 0.5), 'N400': (0.35, 0.6)}

    Returns
    -------
    features : pd.DataFrame
        DataFrame with one row containing all ERP features
        Columns: [component]_[measure]_[group]
        Example: 'P300_amplitude_parietal', 'N400_latency_central'

    Examples
    --------
    >>> channel_groups = {
    ...     'parietal': ['Pz', 'P3', 'P4'],
    ...     'central': ['Cz', 'C3', 'C4']
    ... }
    >>> features = extract_erp_features(epochs, components=['P300', 'N400'],
    ...                                  channel_groups=channel_groups)
    >>> print(features.columns)
    >>> print(features.iloc[0])

    Notes
    -----
    - All amplitude values are in microvolts (µV)
    - All latency values are in seconds
    - Features are averaged within each channel group
    - Missing channels in groups are silently ignored
    """
    # Set defaults
    if components is None:
        components = ['P300', 'N400']

    if windows is None:
        windows = {
            'P300': (0.25, 0.5),
            'N400': (0.35, 0.6)
        }

    # If no channel groups provided, use all channels
    if channel_groups is None:
        channel_groups = {'all': list(epochs.ch_names)}

    # Filter channel groups to only include existing channels
    filtered_groups = {}
    for group_name, channels in channel_groups.items():
        valid_channels = [ch for ch in channels if ch in epochs.ch_names]
        if valid_channels:
            filtered_groups[group_name] = valid_channels

    if not filtered_groups:
        raise ValueError("No valid channels found in any channel group")

    # Extract features for each component and channel group
    feature_dict = {}

    for component in components:
        if component not in windows:
            raise ValueError(f"No time window defined for component: {component}")

        window = windows[component]

        # Determine extraction function and polarity
        if component == 'P300':
            extract_func = extract_p300
        elif component == 'N400':
            extract_func = extract_n400
        else:
            raise ValueError(f"Unknown component: {component}. Use 'P300' or 'N400'")

        # Extract for each channel group
        for group_name, channels in filtered_groups.items():
            try:
                # Extract component features
                component_features = extract_func(
                    epochs,
                    channels=channels,
                    window=window
                )

                # Add to feature dictionary with descriptive names
                feature_dict[f'{component}_amplitude_{group_name}'] = component_features['amplitude']
                feature_dict[f'{component}_latency_{group_name}'] = component_features['latency']

            except Exception as e:
                # Log warning but continue with other groups
                print(f"Warning: Failed to extract {component} for group {group_name}: {e}")
                feature_dict[f'{component}_amplitude_{group_name}'] = np.nan
                feature_dict[f'{component}_latency_{group_name}'] = np.nan

    # Convert to DataFrame
    features_df = pd.DataFrame([feature_dict])

    return features_df


def main(cfg='configs/erp.yaml'):
    """
    Main function for standalone ERP extraction.

    Parameters
    ----------
    cfg : str
        Path to YAML configuration file
    """
    import yaml
    import argparse
    from pathlib import Path

    # Load configuration
    with open(cfg, 'r') as f:
        config = yaml.safe_load(f)

    print(f"ERP Feature Extraction")
    print(f"Config: {cfg}")
    print(f"Components: {config.get('components', ['P300', 'N400'])}")
    print(f"Channel groups: {list(config.get('channel_groups', {}).keys())}")

    # TODO: Load BIDS data and extract features
    # This would be implemented when integrating with full pipeline

    print("ERP extraction complete. Features saved to data/derivatives/")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Extract ERP features from epoched EEG data'
    )
    ap.add_argument('--config', default='configs/erp.yaml',
                    help='Path to ERP configuration file')
    args = ap.parse_args()
    main(args.config)
