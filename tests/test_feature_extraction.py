"""
Test suite for feature extraction modules following TDD principles.
Tests all 4 feature families: Spectrum, Complexity, Connectivity, ERP
"""
import pytest
import numpy as np
import mne
from pathlib import Path
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.features.spectrum import (
    compute_psd,
    compute_relative_power,
    compute_peak_frequency,
    extract_spectral_features
)
from src.features.entropy import (
    compute_permutation_entropy,
    compute_sample_entropy,
    compute_shannon_entropy,
    compute_lempel_ziv_complexity,
    compute_fractal_dimension,
    extract_complexity_features
)
from src.features.connectivity import (
    compute_correlation,
    compute_aec,
    compute_pli,
    compute_plv,
    compute_dtf,
    compute_granger_causality,
    compute_graph_metrics,
    extract_connectivity_features
)
from src.features.erp import (
    extract_p300,
    extract_n400,
    compute_erp_amplitude,
    compute_erp_latency,
    extract_erp_features
)


class TestSpectrumFeatures:
    """Test spectral feature extraction"""

    @pytest.fixture
    def mock_data(self):
        """Create mock EEG data with known frequency content"""
        sfreq = 256
        duration = 10  # seconds
        n_channels = 64
        times = np.arange(0, duration, 1/sfreq)

        # Create data with specific frequency components
        data = np.zeros((n_channels, len(times)))
        for ch in range(n_channels):
            # Add known frequency components
            data[ch] = (
                0.5 * np.sin(2 * np.pi * 10 * times) +  # 10 Hz (alpha)
                0.3 * np.sin(2 * np.pi * 20 * times) +  # 20 Hz (beta)
                0.2 * np.sin(2 * np.pi * 5 * times) +   # 5 Hz (theta)
                0.1 * np.random.randn(len(times))       # noise
            ) * 1e-6

        info = mne.create_info(
            ch_names=[f'EEG{i:03d}' for i in range(1, n_channels+1)],
            sfreq=sfreq,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(data, info)
        # Skip montage setting as it's not needed for spectral analysis
        # and test channels don't match standard montage names
        return raw

    def test_compute_psd(self, mock_data):
        """Test power spectral density computation"""
        freqs, psd = compute_psd(
            mock_data,
            fmin=1,
            fmax=40,
            method='welch'
        )

        assert len(freqs) > 0
        assert psd.shape[0] == len(mock_data.ch_names)
        assert psd.shape[1] == len(freqs)

        # Check that peak is around 10 Hz
        peak_freq_idx = np.argmax(np.mean(psd, axis=0))
        peak_freq = freqs[peak_freq_idx]
        assert 8 < peak_freq < 12  # Should be around 10 Hz

    def test_compute_relative_power(self, mock_data):
        """Test relative power computation in frequency bands"""
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }

        rel_power = compute_relative_power(mock_data, bands)

        assert 'alpha' in rel_power
        assert 'theta' in rel_power
        assert len(rel_power['alpha']) == len(mock_data.ch_names)

        # Alpha should have highest relative power (10 Hz component)
        avg_powers = {band: np.mean(power) for band, power in rel_power.items()}
        assert max(avg_powers, key=avg_powers.get) == 'alpha'

    def test_compute_peak_frequency(self, mock_data):
        """Test peak frequency detection"""
        peak_freqs = compute_peak_frequency(
            mock_data,
            fmin=8,
            fmax=13  # Alpha band
        )

        assert len(peak_freqs) == len(mock_data.ch_names)
        assert all(8 <= f <= 13 for f in peak_freqs)

        # Should detect 10 Hz peak
        mean_peak = np.mean(peak_freqs)
        assert 9 < mean_peak < 11

    def test_extract_spectral_features(self, mock_data):
        """Test complete spectral feature extraction"""
        features = extract_spectral_features(
            mock_data,
            bands={
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30)
            }
        )

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(mock_data.ch_names)

        # Check feature columns exist
        expected_cols = [
            'theta_power', 'alpha_power', 'beta_power',
            'theta_rel_power', 'alpha_rel_power', 'beta_rel_power',
            'peak_freq_alpha', 'peak_freq_beta'
        ]
        for col in expected_cols:
            assert col in features.columns


class TestEntropyFeatures:
    """Test entropy and complexity feature extraction"""

    @pytest.fixture
    def test_signal(self):
        """Create test signals with known complexity"""
        sfreq = 256
        duration = 2
        times = np.arange(0, duration, 1/sfreq)

        # Simple sinusoid (low complexity)
        simple = np.sin(2 * np.pi * 10 * times)

        # Complex signal (high complexity)
        complex_sig = (
            np.sin(2 * np.pi * 10 * times) +
            0.5 * np.sin(2 * np.pi * 23 * times) +
            0.3 * np.sin(2 * np.pi * 37 * times) +
            0.2 * np.random.randn(len(times))
        )

        # Random signal (maximum complexity)
        random = np.random.randn(len(times))

        return simple, complex_sig, random, sfreq

    def test_permutation_entropy(self, test_signal):
        """Test permutation entropy calculation"""
        simple, complex_sig, random, _ = test_signal

        pe_simple = compute_permutation_entropy(simple, order=3, delay=1)
        pe_complex = compute_permutation_entropy(complex_sig, order=3, delay=1)
        pe_random = compute_permutation_entropy(random, order=3, delay=1)

        # Entropy should increase with complexity
        assert pe_simple < pe_complex < pe_random
        assert 0 <= pe_simple <= 1
        assert 0 <= pe_complex <= 1
        assert 0 <= pe_random <= 1

    def test_sample_entropy(self, test_signal):
        """Test sample entropy calculation"""
        simple, complex_sig, random, _ = test_signal

        se_simple = compute_sample_entropy(simple[:500], m=2, r=0.2)
        se_complex = compute_sample_entropy(complex_sig[:500], m=2, r=0.2)
        se_random = compute_sample_entropy(random[:500], m=2, r=0.2)

        # Sample entropy should increase with irregularity
        assert se_simple < se_complex
        assert all(e >= 0 for e in [se_simple, se_complex, se_random])

    def test_shannon_entropy(self, test_signal):
        """Test Shannon entropy calculation"""
        simple, complex_sig, random, _ = test_signal

        she_simple = compute_shannon_entropy(simple)
        she_complex = compute_shannon_entropy(complex_sig)
        she_random = compute_shannon_entropy(random)

        assert she_simple < she_random
        assert all(e >= 0 for e in [she_simple, she_complex, she_random])

    def test_lempel_ziv_complexity(self, test_signal):
        """Test Lempel-Ziv complexity"""
        simple, complex_sig, random, _ = test_signal

        lzc_simple = compute_lempel_ziv_complexity(simple)
        lzc_complex = compute_lempel_ziv_complexity(complex_sig)
        lzc_random = compute_lempel_ziv_complexity(random)

        # LZC should increase with randomness
        assert lzc_simple < lzc_complex < lzc_random

    def test_fractal_dimension(self, test_signal):
        """Test fractal dimension calculation"""
        simple, complex_sig, random, _ = test_signal

        fd_simple = compute_fractal_dimension(simple)
        fd_complex = compute_fractal_dimension(complex_sig)
        fd_random = compute_fractal_dimension(random)

        # Fractal dimension between 1 and 2 for time series
        assert 1 < fd_simple < 2
        assert 1 < fd_complex < 2
        assert 1 < fd_random < 2

        # More complex signals have higher FD
        assert fd_simple < fd_random


class TestConnectivityFeatures:
    """Test connectivity feature extraction"""

    @pytest.fixture
    def mock_epochs(self):
        """Create mock epoched data"""
        sfreq = 256
        n_epochs = 20
        n_channels = 10
        n_times = 256  # 1 second epochs

        # Create correlated signals
        data = np.zeros((n_epochs, n_channels, n_times))
        for epoch in range(n_epochs):
            # Base signal
            base = np.random.randn(n_times)

            # Create channels with varying correlation
            for ch in range(n_channels):
                if ch < 5:
                    # High correlation group
                    data[epoch, ch] = base + 0.1 * np.random.randn(n_times)
                else:
                    # Low correlation group
                    data[epoch, ch] = np.random.randn(n_times)

        info = mne.create_info(
            ch_names=[f'EEG{i:03d}' for i in range(1, n_channels+1)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        epochs = mne.EpochsArray(data * 1e-6, info)
        return epochs

    def test_compute_correlation(self, mock_epochs):
        """Test correlation connectivity"""
        corr_matrix = compute_correlation(mock_epochs)

        assert corr_matrix.shape == (len(mock_epochs.ch_names), len(mock_epochs.ch_names))
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Self-correlation = 1

        # Check symmetry
        assert np.allclose(corr_matrix, corr_matrix.T)

        # High correlation within first 5 channels
        within_group = corr_matrix[:5, :5]
        between_group = corr_matrix[:5, 5:]
        assert np.mean(within_group) > np.mean(between_group)

    def test_compute_pli(self, mock_epochs):
        """Test Phase Lag Index"""
        pli_matrix = compute_pli(mock_epochs, fmin=8, fmax=13)

        assert pli_matrix.shape == (len(mock_epochs.ch_names), len(mock_epochs.ch_names))
        assert np.all(pli_matrix >= 0) and np.all(pli_matrix <= 1)
        assert np.allclose(np.diag(pli_matrix), 0)  # No self-connectivity

    def test_compute_plv(self, mock_epochs):
        """Test Phase Locking Value"""
        plv_matrix = compute_plv(mock_epochs, fmin=8, fmax=13)

        assert plv_matrix.shape == (len(mock_epochs.ch_names), len(mock_epochs.ch_names))
        assert np.all(plv_matrix >= 0) and np.all(plv_matrix <= 1)
        assert np.allclose(np.diag(plv_matrix), 1)  # Perfect self-locking

    def test_compute_graph_metrics(self, mock_epochs):
        """Test graph theory metrics"""
        # Compute connectivity matrix
        conn_matrix = compute_correlation(mock_epochs)

        # Compute graph metrics
        metrics = compute_graph_metrics(
            conn_matrix,
            threshold=0.3,
            metrics=['clustering_coeff', 'path_length', 'small_world']
        )

        assert 'clustering_coeff' in metrics
        assert 'path_length' in metrics
        assert 'small_world' in metrics

        assert 0 <= metrics['clustering_coeff'] <= 1
        assert metrics['path_length'] > 0
        assert metrics['small_world'] > 0


class TestERPFeatures:
    """Test ERP feature extraction"""

    @pytest.fixture
    def mock_erp_data(self):
        """Create mock ERP data with P300 and N400 components"""
        sfreq = 256
        n_epochs = 50
        n_channels = 64
        tmin = -0.2
        tmax = 0.8

        times = np.linspace(tmin, tmax, int(sfreq * (tmax - tmin)))

        # Create ERP components
        data = np.zeros((n_epochs, n_channels, len(times)))

        for epoch in range(n_epochs):
            for ch in range(n_channels):
                # Add P300 (positive peak around 300ms)
                p300_latency = 0.3 + np.random.normal(0, 0.02)
                p300_amplitude = 5 + np.random.normal(0, 1)

                # Add N400 (negative peak around 400ms)
                n400_latency = 0.4 + np.random.normal(0, 0.02)
                n400_amplitude = -3 + np.random.normal(0, 0.5)

                # Gaussian-shaped components
                data[epoch, ch] += p300_amplitude * np.exp(
                    -((times - p300_latency) ** 2) / (2 * 0.05 ** 2)
                )
                data[epoch, ch] += n400_amplitude * np.exp(
                    -((times - n400_latency) ** 2) / (2 * 0.05 ** 2)
                )

                # Add noise
                data[epoch, ch] += np.random.randn(len(times)) * 0.5

        info = mne.create_info(
            ch_names=[f'EEG{i:03d}' for i in range(1, n_channels+1)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        epochs = mne.EpochsArray(data * 1e-6, info, tmin=tmin)
        return epochs

    def test_extract_p300(self, mock_erp_data):
        """Test P300 extraction"""
        p300_features = extract_p300(
            mock_erp_data,
            channels='all',
            window=(0.25, 0.5)
        )

        assert 'amplitude' in p300_features
        assert 'latency' in p300_features

        # Check P300 characteristics
        assert p300_features['amplitude'] > 0  # Positive component
        assert 0.25 < p300_features['latency'] < 0.5  # Within window

    def test_extract_n400(self, mock_erp_data):
        """Test N400 extraction"""
        n400_features = extract_n400(
            mock_erp_data,
            channels='all',
            window=(0.35, 0.6)
        )

        assert 'amplitude' in n400_features
        assert 'latency' in n400_features

        # Check N400 characteristics
        assert n400_features['amplitude'] < 0  # Negative component
        assert 0.35 < n400_features['latency'] < 0.6  # Within window

    def test_compute_erp_amplitude(self, mock_erp_data):
        """Test ERP amplitude computation"""
        amplitudes = compute_erp_amplitude(
            mock_erp_data,
            time_window=(0.25, 0.35)
        )

        assert len(amplitudes) == len(mock_erp_data.ch_names)
        assert all(isinstance(a, (int, float)) for a in amplitudes)

    def test_compute_erp_latency(self, mock_erp_data):
        """Test ERP latency computation"""
        latencies = compute_erp_latency(
            mock_erp_data,
            time_window=(0.25, 0.5),
            polarity='positive'
        )

        assert len(latencies) == len(mock_erp_data.ch_names)
        assert all(0.25 <= l <= 0.5 for l in latencies)

    def test_extract_erp_features(self, mock_erp_data):
        """Test complete ERP feature extraction"""
        features = extract_erp_features(
            mock_erp_data,
            components=['P300', 'N400'],
            channel_groups={
                'parietal': ['EEG030', 'EEG031', 'EEG032'],
                'central': ['EEG020', 'EEG021', 'EEG022']
            }
        )

        assert isinstance(features, pd.DataFrame)
        assert 'P300_amplitude_parietal' in features.columns
        assert 'P300_latency_parietal' in features.columns
        assert 'N400_amplitude_central' in features.columns
        assert 'N400_latency_central' in features.columns