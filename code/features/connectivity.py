#!/usr/bin/env python
"""
Connectivity feature extraction for EEG data
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import mne
# from mne.connectivity import spectral_connectivity_epochs
import yaml
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_plv(signals: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute Phase Locking Value connectivity

    Parameters
    ----------
    signals : array (n_channels, n_samples)
        Input signals
    sfreq : float
        Sampling frequency

    Returns
    -------
    plv : array (n_channels, n_channels)
        PLV connectivity matrix
    """
    n_channels, n_samples = signals.shape

    # Compute Hilbert transform for phase
    from scipy.signal import hilbert
    analytic_signals = hilbert(signals, axis=1)
    phases = np.angle(analytic_signals)

    # Compute PLV
    plv = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            phase_diff = phases[i] - phases[j]
            plv[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv[j, i] = plv[i, j]

    np.fill_diagonal(plv, 1.0)
    return plv

def compute_pli(signals: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute Phase Lag Index connectivity

    Parameters
    ----------
    signals : array (n_channels, n_samples)
        Input signals
    sfreq : float
        Sampling frequency

    Returns
    -------
    pli : array (n_channels, n_channels)
        PLI connectivity matrix
    """
    n_channels, n_samples = signals.shape

    # Compute Hilbert transform for phase
    from scipy.signal import hilbert
    analytic_signals = hilbert(signals, axis=1)
    phases = np.angle(analytic_signals)

    # Compute PLI
    pli = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            phase_diff = phases[i] - phases[j]
            pli[i, j] = np.abs(np.mean(np.sign(np.sin(phase_diff))))
            pli[j, i] = pli[i, j]

    return pli

def compute_aec(signals: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute Amplitude Envelope Correlation

    Parameters
    ----------
    signals : array (n_channels, n_samples)
        Input signals
    sfreq : float
        Sampling frequency

    Returns
    -------
    aec : array (n_channels, n_channels)
        AEC connectivity matrix
    """
    n_channels, n_samples = signals.shape

    # Compute amplitude envelopes
    from scipy.signal import hilbert
    analytic_signals = hilbert(signals, axis=1)
    envelopes = np.abs(analytic_signals)

    # Compute correlations
    aec = np.corrcoef(envelopes)

    return aec

def compute_graph_metrics(connectivity_matrix: np.ndarray) -> Dict:
    """
    Compute graph theory metrics from connectivity matrix

    Parameters
    ----------
    connectivity_matrix : array (n_channels, n_channels)
        Connectivity matrix

    Returns
    -------
    metrics : dict
        Graph theory metrics
    """
    # Threshold to create binary adjacency matrix
    threshold = np.percentile(connectivity_matrix[~np.eye(connectivity_matrix.shape[0], dtype=bool)], 75)
    adj_matrix = (connectivity_matrix > threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)

    n_nodes = adj_matrix.shape[0]
    metrics = {}

    # Clustering coefficient
    clustering_coeffs = []
    for i in range(n_nodes):
        neighbors = np.where(adj_matrix[i])[0]
        if len(neighbors) > 1:
            subgraph = adj_matrix[np.ix_(neighbors, neighbors)]
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            actual_edges = np.sum(subgraph) / 2
            clustering_coeffs.append(actual_edges / possible_edges if possible_edges > 0 else 0)
        else:
            clustering_coeffs.append(0)

    metrics['clustering'] = np.mean(clustering_coeffs)

    # Average path length (simplified)
    # Using Floyd-Warshall algorithm
    dist = adj_matrix.copy().astype(float)
    dist[dist == 0] = np.inf
    np.fill_diagonal(dist, 0)

    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    # Remove infinite values (disconnected nodes)
    valid_paths = dist[np.isfinite(dist) & (dist > 0)]
    metrics['path_length'] = np.mean(valid_paths) if len(valid_paths) > 0 else 0

    # Global efficiency
    efficiency_vals = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if dist[i, j] != np.inf and dist[i, j] > 0:
                efficiency_vals.append(1 / dist[i, j])

    metrics['efficiency'] = np.mean(efficiency_vals) if efficiency_vals else 0

    # Modularity (simplified - degree-based)
    degrees = np.sum(adj_matrix, axis=1)
    total_edges = np.sum(adj_matrix) / 2
    if total_edges > 0:
        modularity = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j]:
                    expected = (degrees[i] * degrees[j]) / (2 * total_edges)
                    modularity += adj_matrix[i, j] - expected
        metrics['modularity'] = modularity / (2 * total_edges)
    else:
        metrics['modularity'] = 0

    return metrics

def extract_connectivity_features(raw: mne.io.Raw, config: Dict) -> Dict:
    """
    Extract connectivity features from EEG data

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed EEG data
    config : dict
        Configuration parameters

    Returns
    -------
    features : dict
        Dictionary of connectivity features
    """
    features = {}

    # Get EEG channels
    picks = mne.pick_types(raw.info, meg=False, eeg=True)
    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[i] for i in picks]
    sfreq = raw.info['sfreq']

    # Get connectivity measures from config
    measures = config.get('connectivity_measures', ['correlation', 'plv', 'pli'])

    # Compute connectivity matrices for each measure
    connectivity_matrices = {}

    # Correlation
    if 'correlation' in measures:
        corr_matrix = np.corrcoef(data)
        connectivity_matrices['correlation'] = corr_matrix
        # Extract upper triangle values (excluding diagonal)
        upper_tri_indices = np.triu_indices(len(ch_names), k=1)
        corr_values = corr_matrix[upper_tri_indices]
        features['conn_corr_mean'] = np.mean(corr_values)
        features['conn_corr_std'] = np.std(corr_values)
        features['conn_corr_max'] = np.max(corr_values)

    # PLV
    if 'plv' in measures:
        plv_matrix = compute_plv(data, sfreq)
        connectivity_matrices['plv'] = plv_matrix
        upper_tri_indices = np.triu_indices(len(ch_names), k=1)
        plv_values = plv_matrix[upper_tri_indices]
        features['conn_plv_mean'] = np.mean(plv_values)
        features['conn_plv_std'] = np.std(plv_values)
        features['conn_plv_max'] = np.max(plv_values)

    # PLI
    if 'pli' in measures:
        pli_matrix = compute_pli(data, sfreq)
        connectivity_matrices['pli'] = pli_matrix
        upper_tri_indices = np.triu_indices(len(ch_names), k=1)
        pli_values = pli_matrix[upper_tri_indices]
        features['conn_pli_mean'] = np.mean(pli_values)
        features['conn_pli_std'] = np.std(pli_values)
        features['conn_pli_max'] = np.max(pli_values)

    # AEC
    if 'aec' in measures:
        aec_matrix = compute_aec(data, sfreq)
        connectivity_matrices['aec'] = aec_matrix
        upper_tri_indices = np.triu_indices(len(ch_names), k=1)
        aec_values = aec_matrix[upper_tri_indices]
        features['conn_aec_mean'] = np.mean(aec_values)
        features['conn_aec_std'] = np.std(aec_values)
        features['conn_aec_max'] = np.max(aec_values)

    # Compute graph metrics for each connectivity matrix
    graph_metrics_list = config.get('graph_metrics', ['clustering', 'path_length', 'efficiency'])

    for measure_name, matrix in connectivity_matrices.items():
        if len(graph_metrics_list) > 0:
            graph_metrics = compute_graph_metrics(matrix)
            for metric_name in graph_metrics_list:
                if metric_name in graph_metrics:
                    features[f'graph_{measure_name}_{metric_name}'] = graph_metrics[metric_name]

    return features

def main():
    parser = argparse.ArgumentParser(description='Extract connectivity features from EEG data')
    parser.add_argument('--input_dir', type=str, default='data/derivatives/clean',
                       help='Directory with preprocessed EEG files')
    parser.add_argument('--config', type=str, default='configs/connectivity.yaml',
                       help='Configuration file')
    parser.add_argument('--output_file', type=str, default='data/derivatives/features_connectivity.parquet',
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
            features = extract_connectivity_features(raw, config)
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
    print(f"  - Connectivity measures: {sum('conn_' in c for c in df.columns)}")
    print(f"  - Graph metrics: {sum('graph_' in c for c in df.columns)}")

if __name__ == '__main__':
    main()
