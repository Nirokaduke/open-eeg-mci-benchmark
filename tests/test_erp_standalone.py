"""
Standalone test for ERP feature extraction module.

Tests all ERP functions independently without requiring other feature modules.
"""
import pytest
import numpy as np
import mne
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.features.erp import (
    extract_p300,
    extract_n400,
    compute_erp_amplitude,
    compute_erp_latency,
    extract_erp_features
)


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

        print(f"P300 - Amplitude: {p300_features['amplitude']:.2f} uV, "
              f"Latency: {p300_features['latency']*1000:.0f} ms")

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

        print(f"N400 - Amplitude: {n400_features['amplitude']:.2f} uV, "
              f"Latency: {n400_features['latency']*1000:.0f} ms")

    def test_compute_erp_amplitude(self, mock_erp_data):
        """Test ERP amplitude computation"""
        amplitudes = compute_erp_amplitude(
            mock_erp_data,
            time_window=(0.25, 0.35)
        )

        assert len(amplitudes) == len(mock_erp_data.ch_names)
        assert all(isinstance(a, (int, float, np.number)) for a in amplitudes)

        print(f"ERP amplitudes computed: {len(amplitudes)} channels, "
              f"mean={np.mean(amplitudes):.2f} uV")

    def test_compute_erp_latency(self, mock_erp_data):
        """Test ERP latency computation"""
        latencies = compute_erp_latency(
            mock_erp_data,
            time_window=(0.25, 0.5),
            polarity='positive'
        )

        assert len(latencies) == len(mock_erp_data.ch_names)
        assert all(0.25 <= l <= 0.5 for l in latencies)

        print(f"ERP latencies computed: {len(latencies)} channels, "
              f"mean={np.mean(latencies)*1000:.0f} ms")

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

        print(f"\nExtracted ERP features:")
        print(features.to_string())

    def test_extract_erp_features_all_channels(self, mock_erp_data):
        """Test ERP feature extraction with all channels"""
        features = extract_erp_features(
            mock_erp_data,
            components=['P300', 'N400']
        )

        assert isinstance(features, pd.DataFrame)
        assert 'P300_amplitude_all' in features.columns
        assert 'P300_latency_all' in features.columns
        assert 'N400_amplitude_all' in features.columns
        assert 'N400_latency_all' in features.columns

        # Check that P300 is positive and N400 is negative
        assert features['P300_amplitude_all'].iloc[0] > 0
        assert features['N400_amplitude_all'].iloc[0] < 0

        print(f"\nAll-channel ERP features:")
        print(features.to_string())


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
