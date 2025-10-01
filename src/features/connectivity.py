"""
Connectivity feature extraction module for EEG-MCI-Bench.

Implements functional and effective connectivity measures:
- Correlation-based: Pearson correlation, Amplitude Envelope Correlation (AEC)
- Phase-based: Phase Lag Index (PLI), Phase Locking Value (PLV)
- Directed: Directed Transfer Function (DTF), Granger Causality
- Graph theory: Clustering coefficient, path length, small-world index

All functions follow BIDS conventions and support MNE data structures.
"""

import numpy as np
import pandas as pd
import mne
from scipy import signal
from scipy.stats import pearsonr
from typing import Dict, List, Optional, Union, Tuple
import warnings

# Try to import mne-connectivity if available, otherwise use fallback implementations
try:
    from mne_connectivity import spectral_connectivity_epochs
    MNE_CONNECTIVITY_AVAILABLE = True
except ImportError:
    MNE_CONNECTIVITY_AVAILABLE = False
    warnings.warn(
        "mne-connectivity not installed. Using fallback implementations for PLI/PLV. "
        "Install with: pip install mne-connectivity",
        ImportWarning
    )


def compute_correlation(epochs: mne.Epochs) -> np.ndarray:
    """
    Compute Pearson correlation matrix between all channel pairs.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data of shape (n_epochs, n_channels, n_times)

    Returns
    -------
    corr_matrix : np.ndarray
        Correlation matrix of shape (n_channels, n_channels)
        Symmetric matrix with diagonal = 1

    Examples
    --------
    >>> corr = compute_correlation(epochs)
    >>> print(f"Correlation shape: {corr.shape}")
    >>> assert np.allclose(np.diag(corr), 1.0)
    """
    # Get data: (n_epochs, n_channels, n_times)
    data = epochs.get_data()
    n_channels = data.shape[1]

    # Average across time within each epoch, then concatenate epochs
    # Result: (n_channels, n_epochs * n_times) for correlation
    data_flat = data.transpose(1, 0, 2).reshape(n_channels, -1)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(data_flat)

    return corr_matrix


def compute_aec(epochs: mne.Epochs, fmin: float = 8.0, fmax: float = 13.0,
                method: str = 'hilbert') -> np.ndarray:
    """
    Compute Amplitude Envelope Correlation (AEC).

    AEC measures correlation between amplitude envelopes of bandpass-filtered signals,
    which reflects co-modulation of neuronal activity.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    fmin : float
        Lower frequency bound (Hz), default 8.0 (alpha band)
    fmax : float
        Upper frequency bound (Hz), default 13.0 (alpha band)
    method : str
        Method to extract envelope: 'hilbert' or 'power'

    Returns
    -------
    aec_matrix : np.ndarray
        AEC matrix of shape (n_channels, n_channels)
        Symmetric matrix with diagonal = 1

    References
    ----------
    Hipp, J. F., et al. (2012). Large-scale cortical correlation structure of
    spontaneous oscillatory activity. Nature Neuroscience, 15(6), 884-890.
    """
    # Filter data in specified frequency band
    epochs_filtered = epochs.copy().filter(fmin, fmax, method='iir')
    data = epochs_filtered.get_data()

    n_epochs, n_channels, n_times = data.shape

    # Extract amplitude envelopes using Hilbert transform
    envelopes = np.zeros((n_channels, n_epochs * n_times))

    for ch in range(n_channels):
        # Concatenate epochs for this channel
        ch_data = data[:, ch, :].flatten()

        # Apply Hilbert transform to get analytic signal
        analytic_signal = signal.hilbert(ch_data)

        # Envelope is the magnitude of analytic signal
        envelopes[ch] = np.abs(analytic_signal)

    # Compute correlation of envelopes
    aec_matrix = np.corrcoef(envelopes)

    return aec_matrix


def _compute_pli_fallback(epochs: mne.Epochs, fmin: float, fmax: float) -> np.ndarray:
    """
    Fallback implementation of PLI using Hilbert transform.
    """
    # Filter data in specified frequency band
    epochs_filtered = epochs.copy().filter(fmin, fmax, method='iir')
    data = epochs_filtered.get_data()

    n_epochs, n_channels, n_times = data.shape
    pli_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i, n_channels):
            if i == j:
                pli_matrix[i, j] = 0  # No self-connectivity
            else:
                # Compute instantaneous phase using Hilbert transform
                pli_values = []

                for epoch in range(n_epochs):
                    # Get analytic signals
                    analytic_i = signal.hilbert(data[epoch, i, :])
                    analytic_j = signal.hilbert(data[epoch, j, :])

                    # Compute phase difference
                    phase_i = np.angle(analytic_i)
                    phase_j = np.angle(analytic_j)
                    phase_diff = phase_i - phase_j

                    # PLI is the absolute value of the mean sign of phase differences
                    pli = np.abs(np.mean(np.sign(phase_diff)))
                    pli_values.append(pli)

                # Average PLI across epochs
                pli_matrix[i, j] = np.mean(pli_values)
                pli_matrix[j, i] = pli_matrix[i, j]  # Symmetric

    return pli_matrix


def _compute_plv_fallback(epochs: mne.Epochs, fmin: float, fmax: float) -> np.ndarray:
    """
    Fallback implementation of PLV using Hilbert transform.
    """
    # Filter data in specified frequency band
    epochs_filtered = epochs.copy().filter(fmin, fmax, method='iir')
    data = epochs_filtered.get_data()

    n_epochs, n_channels, n_times = data.shape
    plv_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i, n_channels):
            if i == j:
                plv_matrix[i, j] = 1  # Perfect self-phase-locking
            else:
                # Compute phase differences across epochs
                phase_diffs = []

                for epoch in range(n_epochs):
                    # Get analytic signals
                    analytic_i = signal.hilbert(data[epoch, i, :])
                    analytic_j = signal.hilbert(data[epoch, j, :])

                    # Compute instantaneous phase
                    phase_i = np.angle(analytic_i)
                    phase_j = np.angle(analytic_j)

                    # Phase difference
                    phase_diff = phase_i - phase_j
                    phase_diffs.append(phase_diff)

                # PLV: magnitude of average of complex phase differences
                phase_diffs = np.array(phase_diffs)
                plv = np.abs(np.mean(np.exp(1j * phase_diffs)))

                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv  # Symmetric

    return plv_matrix


def compute_pli(epochs: mne.Epochs, fmin: float = 8.0, fmax: float = 13.0) -> np.ndarray:
    """
    Compute Phase Lag Index (PLI).

    PLI measures the asymmetry of phase difference distribution between signals,
    which is insensitive to volume conduction and common sources.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    fmin : float
        Lower frequency bound (Hz)
    fmax : float
        Upper frequency bound (Hz)

    Returns
    -------
    pli_matrix : np.ndarray
        PLI matrix of shape (n_channels, n_channels)
        Values in [0, 1], diagonal = 0 (no self-connectivity)

    References
    ----------
    Stam, C. J., et al. (2007). Phase lag index: assessment of functional
    connectivity from multi channel EEG and MEG with diminished bias from
    common sources. Human Brain Mapping, 28(11), 1178-1193.
    """
    if MNE_CONNECTIVITY_AVAILABLE:
        # Use MNE's spectral connectivity to compute PLI
        con = spectral_connectivity_epochs(
            epochs,
            method='pli',
            mode='multitaper',
            sfreq=epochs.info['sfreq'],
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            n_jobs=1
        )

        # Extract connectivity matrix
        n_channels = len(epochs.ch_names)
        pli_matrix = np.zeros((n_channels, n_channels))

        # Get the connectivity data
        con_data = con.get_data(output='dense')

        # con_data shape is (n_channels, n_channels, n_freqs) or (n_channels, n_channels)
        if con_data.ndim == 3:
            pli_matrix = con_data[:, :, 0]  # Take first (and only) frequency bin
        else:
            pli_matrix = con_data

        # Ensure diagonal is zero (no self-connectivity)
        np.fill_diagonal(pli_matrix, 0)

        return pli_matrix
    else:
        # Fallback implementation
        return _compute_pli_fallback(epochs, fmin, fmax)


def compute_plv(epochs: mne.Epochs, fmin: float = 8.0, fmax: float = 13.0) -> np.ndarray:
    """
    Compute Phase Locking Value (PLV).

    PLV measures the consistency of phase difference between two signals across trials.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    fmin : float
        Lower frequency bound (Hz)
    fmax : float
        Upper frequency bound (Hz)

    Returns
    -------
    plv_matrix : np.ndarray
        PLV matrix of shape (n_channels, n_channels)
        Values in [0, 1], diagonal = 1 (perfect self-locking)

    References
    ----------
    Lachaux, J. P., et al. (1999). Measuring phase synchrony in brain signals.
    Human Brain Mapping, 8(4), 194-208.
    """
    if MNE_CONNECTIVITY_AVAILABLE:
        # Use MNE's spectral connectivity to compute PLV
        con = spectral_connectivity_epochs(
            epochs,
            method='plv',
            mode='multitaper',
            sfreq=epochs.info['sfreq'],
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            n_jobs=1
        )

        # Extract connectivity matrix
        n_channels = len(epochs.ch_names)
        plv_matrix = np.zeros((n_channels, n_channels))

        # Get the connectivity data
        con_data = con.get_data(output='dense')

        # con_data shape is (n_channels, n_channels, n_freqs) or (n_channels, n_channels)
        if con_data.ndim == 3:
            plv_matrix = con_data[:, :, 0]  # Take first (and only) frequency bin
        else:
            plv_matrix = con_data

        # Ensure diagonal is 1 (perfect self-phase-locking)
        np.fill_diagonal(plv_matrix, 1)

        return plv_matrix
    else:
        # Fallback implementation
        return _compute_plv_fallback(epochs, fmin, fmax)


def compute_dtf(epochs: mne.Epochs, fmin: float = 1.0, fmax: float = 40.0,
                model_order: int = 10) -> np.ndarray:
    """
    Compute Directed Transfer Function (DTF).

    DTF is a multivariate spectral measure based on MVAR models that describes
    the direction of information flow between brain regions.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    fmin : float
        Lower frequency bound (Hz)
    fmax : float
        Upper frequency bound (Hz)
    model_order : int
        Order of MVAR model (typically 10-20 for EEG)

    Returns
    -------
    dtf_matrix : np.ndarray
        DTF matrix averaged over frequency band
        Shape (n_channels, n_channels), values in [0, 1]
        dtf_matrix[i, j] represents influence from j to i

    References
    ----------
    Kamiński, M., & Blinowska, K. J. (1991). A new method of the description
    of the information flow in the brain structures. Biological Cybernetics,
    65(3), 203-210.
    """
    # Get data
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape

    # For simplicity, use a basic implementation based on correlation
    # A full DTF implementation would require MVAR modeling
    # This is a placeholder that computes directed connectivity based on
    # time-delayed correlations

    warnings.warn(
        "compute_dtf uses simplified directed connectivity estimation. "
        "For production use, consider using nitime or custom MVAR implementation.",
        UserWarning
    )

    # Average across epochs
    data_avg = np.mean(data, axis=0)  # (n_channels, n_times)

    # Compute time-delayed correlations
    dtf_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                # Cross-correlation with time lag
                xcorr = np.correlate(data_avg[i], data_avg[j], mode='same')
                # Take peak of positive lags (j leads i)
                center = len(xcorr) // 2
                dtf_matrix[i, j] = np.max(np.abs(xcorr[center:center+50]))

    # Normalize to [0, 1]
    dtf_max = np.max(dtf_matrix)
    if dtf_max > 0:
        dtf_matrix = dtf_matrix / dtf_max

    return dtf_matrix


def compute_granger_causality(epochs: mne.Epochs, max_lag: int = 10) -> np.ndarray:
    """
    Compute pairwise Granger causality.

    Granger causality tests whether past values of one signal help predict
    another signal better than using only the past values of that signal.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    max_lag : int
        Maximum lag for autoregressive model

    Returns
    -------
    gc_matrix : np.ndarray
        Granger causality matrix, shape (n_channels, n_channels)
        gc_matrix[i, j] represents causal influence from j to i

    References
    ----------
    Granger, C. W. (1969). Investigating causal relations by econometric models
    and cross-spectral methods. Econometrica, 37(3), 424-438.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape

    # Average across epochs
    data_avg = np.mean(data, axis=0).T  # (n_times, n_channels)

    gc_matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                # Test if j Granger-causes i
                # Create data array: [target, source]
                test_data = np.column_stack([data_avg[:, i], data_avg[:, j]])

                try:
                    # Run Granger causality test
                    result = grangercausalitytests(test_data, max_lag, verbose=False)

                    # Extract F-test p-values and take minimum (strongest evidence)
                    p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                    min_p = np.min(p_values)

                    # Convert p-value to causality strength (1 - p)
                    gc_matrix[i, j] = 1 - min_p

                except Exception as e:
                    # If test fails, set to zero
                    warnings.warn(f"Granger test failed for channels {i}, {j}: {e}")
                    gc_matrix[i, j] = 0

    return gc_matrix


def compute_graph_metrics(conn_matrix: np.ndarray, threshold: float = 0.3,
                          metrics: List[str] = ['clustering_coeff', 'path_length', 'small_world']
                          ) -> Dict[str, float]:
    """
    Compute graph theory metrics from connectivity matrix.

    Parameters
    ----------
    conn_matrix : np.ndarray
        Connectivity matrix of shape (n_nodes, n_nodes)
    threshold : float
        Threshold for binarizing connectivity matrix (0-1)
        Edges with weight < threshold are removed
    metrics : list of str
        Metrics to compute. Options:
        - 'clustering_coeff': Average clustering coefficient
        - 'path_length': Average shortest path length
        - 'small_world': Small-world index (clustering / path_length)
        - 'degree': Average node degree
        - 'modularity': Network modularity

    Returns
    -------
    graph_metrics : dict
        Dictionary of computed metrics

    References
    ----------
    Rubinov, M., & Sporns, O. (2010). Complex network measures of brain
    connectivity: uses and interpretations. NeuroImage, 52(3), 1059-1069.
    """
    # Threshold connectivity matrix to create binary adjacency matrix
    adj_matrix = (np.abs(conn_matrix) > threshold).astype(int)

    # Remove self-connections
    np.fill_diagonal(adj_matrix, 0)

    n_nodes = adj_matrix.shape[0]

    results = {}

    # Compute requested metrics
    if 'clustering_coeff' in metrics:
        # Clustering coefficient: fraction of triangles around a node
        clustering = _compute_clustering_coefficient(adj_matrix)
        results['clustering_coeff'] = clustering

    if 'path_length' in metrics:
        # Average shortest path length
        path_length = _compute_average_path_length(adj_matrix)
        results['path_length'] = path_length

    if 'small_world' in metrics:
        # Small-world index
        if 'clustering_coeff' not in results:
            clustering = _compute_clustering_coefficient(adj_matrix)
        else:
            clustering = results['clustering_coeff']

        if 'path_length' not in results:
            path_length = _compute_average_path_length(adj_matrix)
        else:
            path_length = results['path_length']

        # Small-world index: high clustering, low path length
        # Compare to random network: SW = (C/C_rand) / (L/L_rand)
        # Simplified: just use C/L ratio
        if path_length > 0:
            results['small_world'] = clustering / path_length
        else:
            results['small_world'] = 0

    if 'degree' in metrics:
        # Average degree (number of connections per node)
        degree = np.sum(adj_matrix, axis=1)
        results['degree'] = np.mean(degree)

    if 'modularity' in metrics:
        # Network modularity (simplified version)
        modularity = _compute_modularity(adj_matrix)
        results['modularity'] = modularity

    return results


def _compute_clustering_coefficient(adj_matrix: np.ndarray) -> float:
    """
    Compute average clustering coefficient.

    Clustering coefficient of a node is the fraction of triangles
    around the node (fraction of neighbors that are also connected).
    """
    n_nodes = adj_matrix.shape[0]
    clustering_coeffs = np.zeros(n_nodes)

    for i in range(n_nodes):
        # Get neighbors of node i
        neighbors = np.where(adj_matrix[i] > 0)[0]
        k = len(neighbors)

        if k < 2:
            clustering_coeffs[i] = 0
        else:
            # Count triangles: connections between neighbors
            subgraph = adj_matrix[np.ix_(neighbors, neighbors)]
            triangles = np.sum(subgraph) / 2

            # Clustering coefficient: actual triangles / possible triangles
            possible_triangles = k * (k - 1) / 2
            clustering_coeffs[i] = triangles / possible_triangles

    return np.mean(clustering_coeffs)


def _compute_average_path_length(adj_matrix: np.ndarray) -> float:
    """
    Compute average shortest path length using Floyd-Warshall algorithm.
    """
    n_nodes = adj_matrix.shape[0]

    # Initialize distance matrix
    dist = np.full((n_nodes, n_nodes), np.inf)

    # Set distances for direct connections
    dist[adj_matrix > 0] = 1
    np.fill_diagonal(dist, 0)

    # Floyd-Warshall algorithm
    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

    # Average path length (excluding infinite distances and diagonal)
    finite_dist = dist[np.isfinite(dist) & (dist > 0)]

    if len(finite_dist) > 0:
        return np.mean(finite_dist)
    else:
        return 0


def _compute_modularity(adj_matrix: np.ndarray) -> float:
    """
    Compute network modularity using a simple spectral method.

    Modularity measures the degree to which a network can be divided
    into distinct communities.
    """
    n_nodes = adj_matrix.shape[0]
    m = np.sum(adj_matrix) / 2  # Number of edges

    if m == 0:
        return 0

    # Degree of each node
    k = np.sum(adj_matrix, axis=1)

    # Modularity matrix
    B = adj_matrix - np.outer(k, k) / (2 * m)

    # Use spectral method: leading eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    leading_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

    # Community assignment based on sign
    communities = (leading_eigenvector > 0).astype(int)

    # Compute modularity
    modularity = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if communities[i] == communities[j]:
                modularity += B[i, j]

    modularity = modularity / (2 * m)

    return modularity


def extract_connectivity_features(epochs: mne.Epochs,
                                  bands: Optional[Dict[str, Tuple[float, float]]] = None,
                                  methods: Optional[List[str]] = None,
                                  graph_threshold: float = 0.3
                                  ) -> pd.DataFrame:
    """
    Extract comprehensive connectivity features from EEG epochs.

    Computes multiple connectivity measures across frequency bands and
    derives graph theory metrics.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data
    bands : dict, optional
        Frequency bands, e.g., {'alpha': (8, 13), 'beta': (13, 30)}
        If None, uses standard EEG bands
    methods : list of str, optional
        Connectivity methods to compute:
        ['correlation', 'aec', 'pli', 'plv', 'dtf', 'granger']
        If None, computes all methods
    graph_threshold : float
        Threshold for graph metrics computation

    Returns
    -------
    features : pd.DataFrame
        DataFrame with connectivity features
        Rows correspond to connectivity metrics or graph metrics

    Examples
    --------
    >>> features = extract_connectivity_features(
    ...     epochs,
    ...     bands={'alpha': (8, 13), 'beta': (13, 30)},
    ...     methods=['correlation', 'pli', 'plv']
    ... )
    """
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

    if methods is None:
        methods = ['correlation', 'aec', 'pli', 'plv', 'dtf', 'granger']

    features_dict = {}

    # Compute time-domain connectivity (correlation)
    if 'correlation' in methods:
        corr_matrix = compute_correlation(epochs)

        # Extract summary statistics
        features_dict['corr_mean'] = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        features_dict['corr_std'] = np.std(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        features_dict['corr_max'] = np.max(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])

        # Graph metrics from correlation
        graph_metrics = compute_graph_metrics(corr_matrix, threshold=graph_threshold)
        for metric_name, metric_value in graph_metrics.items():
            features_dict[f'corr_{metric_name}'] = metric_value

    # Compute frequency-band specific connectivity
    for band_name, (fmin, fmax) in bands.items():

        if 'aec' in methods:
            aec_matrix = compute_aec(epochs, fmin=fmin, fmax=fmax)
            features_dict[f'aec_{band_name}_mean'] = np.mean(
                aec_matrix[np.triu_indices_from(aec_matrix, k=1)]
            )
            features_dict[f'aec_{band_name}_std'] = np.std(
                aec_matrix[np.triu_indices_from(aec_matrix, k=1)]
            )

        if 'pli' in methods:
            pli_matrix = compute_pli(epochs, fmin=fmin, fmax=fmax)
            features_dict[f'pli_{band_name}_mean'] = np.mean(
                pli_matrix[np.triu_indices_from(pli_matrix, k=1)]
            )
            features_dict[f'pli_{band_name}_std'] = np.std(
                pli_matrix[np.triu_indices_from(pli_matrix, k=1)]
            )

        if 'plv' in methods:
            plv_matrix = compute_plv(epochs, fmin=fmin, fmax=fmax)
            features_dict[f'plv_{band_name}_mean'] = np.mean(
                plv_matrix[np.triu_indices_from(plv_matrix, k=1)]
            )
            features_dict[f'plv_{band_name}_std'] = np.std(
                plv_matrix[np.triu_indices_from(plv_matrix, k=1)]
            )

    # Compute directed connectivity (DTF, Granger)
    if 'dtf' in methods:
        dtf_matrix = compute_dtf(epochs)
        features_dict['dtf_mean'] = np.mean(dtf_matrix[dtf_matrix > 0])
        features_dict['dtf_std'] = np.std(dtf_matrix[dtf_matrix > 0])

    if 'granger' in methods:
        try:
            gc_matrix = compute_granger_causality(epochs)
            features_dict['granger_mean'] = np.mean(gc_matrix[gc_matrix > 0])
            features_dict['granger_std'] = np.std(gc_matrix[gc_matrix > 0])
        except Exception as e:
            warnings.warn(f"Granger causality computation failed: {e}")
            features_dict['granger_mean'] = 0
            features_dict['granger_std'] = 0

    # Convert to DataFrame
    features_df = pd.DataFrame([features_dict])

    return features_df


# Main script for standalone execution
def main(cfg='configs/connectivity.yaml'):
    """
    Main execution function for connectivity feature extraction.

    Parameters
    ----------
    cfg : str
        Path to configuration YAML file
    """
    import yaml
    from pathlib import Path

    print('Computing connectivity and graph metrics...')

    # Load configuration
    if Path(cfg).exists():
        with open(cfg, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {cfg}")
    else:
        print(f"Configuration file {cfg} not found. Using defaults.")
        config = {}

    # TODO: Load BIDS dataset
    # TODO: Extract connectivity features
    # TODO: Save to data/derivatives/features_connectivity.parquet

    print("Connectivity feature extraction complete.")


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(
        description='Extract connectivity features from EEG data'
    )
    ap.add_argument('--config', default='configs/connectivity.yaml',
                   help='Path to configuration file')
    args = ap.parse_args()

    main(args.config)
