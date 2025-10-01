"""
Test suite for EEG preprocessing pipeline following TDD principles.
"""
import pytest
import numpy as np
import mne
from pathlib import Path
import yaml
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.preprocessing import (
    load_config,
    apply_filters,
    remove_artifacts,
    re_reference,
    segment_epochs,
    preprocess_raw,
    PreprocessingPipeline
)


class TestPreprocessing:
    """Test preprocessing functionality with configurable parameters"""

    @pytest.fixture
    def mock_raw(self):
        """Create mock EEG data for testing"""
        sfreq = 256
        n_channels = 64
        n_times = sfreq * 60  # 60 seconds

        info = mne.create_info(
            ch_names=[f'EEG{i:03d}' for i in range(1, n_channels+1)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        # Create data with known frequency components
        t = np.arange(n_times) / sfreq
        data = np.zeros((n_channels, n_times))

        # Add different frequency components
        for ch in range(n_channels):
            data[ch] = (
                0.5 * np.sin(2 * np.pi * 10 * t) +  # Alpha
                0.3 * np.sin(2 * np.pi * 20 * t) +  # Beta
                0.2 * np.sin(2 * np.pi * 5 * t) +   # Theta
                np.random.randn(n_times) * 0.1      # Noise
            ) * 1e-6

        raw = mne.io.RawArray(data, info)
        raw.set_montage('standard_1020')
        return raw

    @pytest.fixture
    def config(self):
        """Load preprocessing configuration"""
        config_path = Path('configs/bands.yaml')
        if config_path.exists():
            return yaml.safe_load(open(config_path))
        else:
            # Default config for testing
            return {
                'sampling_rate': 256,
                'highpass_hz': 0.5,
                'lowpass_hz': 40.0,
                'notch_hz': None,
                'bands': {
                    'delta': [1, 4],
                    'theta': [4, 8],
                    'alpha': [8, 13],
                    'beta': [13, 30]
                }
            }

    def test_load_config(self):
        """Test configuration loading"""
        config = load_config('configs/bands.yaml')

        assert 'sampling_rate' in config
        assert 'bands' in config
        assert config['sampling_rate'] == 256
        assert 'alpha' in config['bands']
        assert config['bands']['alpha'] == [8, 13]

    def test_apply_filters(self, mock_raw, config):
        """Test frequency filtering"""
        filtered = apply_filters(
            mock_raw.copy(),
            l_freq=config['highpass_hz'],
            h_freq=config['lowpass_hz'],
            notch_freq=config.get('notch_hz')
        )

        # Check sampling rate unchanged
        assert filtered.info['sfreq'] == mock_raw.info['sfreq']

        # Check filter info is stored
        assert filtered.info.get('highpass', 0) >= config['highpass_hz']
        assert filtered.info.get('lowpass', np.inf) <= config['lowpass_hz']

    def test_artifact_removal_ica(self, mock_raw):
        """Test ICA-based artifact removal"""
        # Add artificial EOG/ECG artifacts
        raw_with_artifacts = mock_raw.copy()

        # Simulate eye blink (high amplitude, low frequency)
        blink_times = [5, 15, 25, 35, 45]  # seconds
        for blink_t in blink_times:
            idx = int(blink_t * raw_with_artifacts.info['sfreq'])
            raw_with_artifacts._data[:10, idx:idx+50] += np.random.randn(10, 50) * 50e-6

        cleaned = remove_artifacts(
            raw_with_artifacts,
            method='ica',
            n_components=15,
            random_state=42
        )

        # Check that high amplitude artifacts are reduced
        original_peak = np.max(np.abs(raw_with_artifacts._data))
        cleaned_peak = np.max(np.abs(cleaned._data))

        assert cleaned_peak < original_peak
        assert cleaned.info['sfreq'] == raw_with_artifacts.info['sfreq']

    def test_artifact_removal_asr(self, mock_raw):
        """Test ASR (Artifact Subspace Reconstruction)"""
        cleaned = remove_artifacts(
            mock_raw.copy(),
            method='asr',
            cutoff=20
        )

        assert cleaned is not None
        assert cleaned.info['nchan'] == mock_raw.info['nchan']

    def test_re_reference(self, mock_raw):
        """Test re-referencing methods"""
        # Test average reference
        avg_ref = re_reference(mock_raw.copy(), ref_type='average')
        assert avg_ref.info['custom_ref_applied'] == mne.io.constants.FIFF.FIFFV_MNE_CUSTOM_REF_ON

        # Test REST reference
        rest_ref = re_reference(mock_raw.copy(), ref_type='rest')
        assert rest_ref is not None

        # Test specific channel reference
        channel_ref = re_reference(
            mock_raw.copy(),
            ref_type='channel',
            ref_channels=['EEG001']
        )
        assert channel_ref is not None

    def test_segment_epochs(self, mock_raw):
        """Test epoch segmentation"""
        # Add events for epoching
        n_events = 10
        event_times = np.linspace(5, 50, n_events)
        events = []

        for event_t in event_times:
            sample = int(event_t * mock_raw.info['sfreq'])
            events.append([sample, 0, 1])

        events = np.array(events)

        epochs = segment_epochs(
            mock_raw,
            events=events,
            event_id={'stimulus': 1},
            tmin=-0.2,
            tmax=0.8,
            baseline=(-0.2, 0)
        )

        assert len(epochs) == n_events
        assert epochs.tmin == -0.2
        assert epochs.tmax == 0.8

    def test_preprocessing_pipeline(self, mock_raw, config):
        """Test complete preprocessing pipeline"""
        pipeline = PreprocessingPipeline(config)

        # Process raw data
        processed = pipeline.process(
            mock_raw.copy(),
            apply_filters=True,
            remove_artifacts=True,
            re_reference=True,
            reference_type='average'
        )

        # Verify all steps were applied
        assert processed.info.get('highpass', 0) >= config['highpass_hz']
        assert processed.info.get('lowpass', np.inf) <= config['lowpass_hz']
        assert processed.info['custom_ref_applied'] == mne.io.constants.FIFF.FIFFV_MNE_CUSTOM_REF_ON

        # Check data integrity
        assert processed.info['nchan'] == mock_raw.info['nchan']
        assert processed.info['sfreq'] == mock_raw.info['sfreq']

    def test_batch_preprocessing(self, mock_raw, config):
        """Test batch processing of multiple subjects"""
        pipeline = PreprocessingPipeline(config)

        # Create multiple mock subjects
        subjects_data = {
            'sub-001': mock_raw.copy(),
            'sub-002': mock_raw.copy(),
            'sub-003': mock_raw.copy()
        }

        processed_data = pipeline.batch_process(
            subjects_data,
            output_dir=Path('data/derivatives/preprocessed'),
            save_intermediate=True
        )

        assert len(processed_data) == 3
        for sub_id, data in processed_data.items():
            assert data is not None
            assert data.info['sfreq'] == config['sampling_rate']

    def test_preprocessing_with_bad_channels(self, mock_raw):
        """Test handling of bad channels"""
        # Mark some channels as bad
        mock_raw.info['bads'] = ['EEG010', 'EEG020', 'EEG030']

        pipeline = PreprocessingPipeline(load_config('configs/bands.yaml'))
        processed = pipeline.process(
            mock_raw.copy(),
            interpolate_bads=True
        )

        # Check bad channels were interpolated
        assert len(processed.info['bads']) == 0

        # Verify channel count is maintained
        assert processed.info['nchan'] == mock_raw.info['nchan']

    def test_save_preprocessed_data(self, mock_raw, tmp_path):
        """Test saving preprocessed data"""
        pipeline = PreprocessingPipeline(load_config('configs/bands.yaml'))

        processed = pipeline.process(mock_raw.copy())

        # Save in BIDS derivatives format
        output_file = tmp_path / 'sub-001_task-rest_proc-preproc_eeg.fif'
        pipeline.save_processed(
            processed,
            output_file,
            save_format='fif'
        )

        assert output_file.exists()

        # Load and verify
        loaded = mne.io.read_raw_fif(output_file, preload=False)
        assert loaded.info['sfreq'] == processed.info['sfreq']
        assert loaded.info['nchan'] == processed.info['nchan']