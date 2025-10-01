"""
EEG Preprocessing Pipeline for EEG-MCI-Bench
Implements filtering, artifact removal, re-referencing, and epoching
"""
import argparse
import mne
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load preprocessing configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        dict: Configuration parameters
    """
    config_path = Path(config_path)
    if not config_path.exists():
        # Return default configuration
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

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def apply_filters(
    raw: mne.io.BaseRaw,
    l_freq: Optional[float] = 0.5,
    h_freq: Optional[float] = 40.0,
    notch_freq: Optional[float] = None,
    verbose: bool = False
) -> mne.io.BaseRaw:
    """
    Apply frequency filtering to raw EEG data

    Args:
        raw: MNE Raw object
        l_freq: High-pass filter frequency (Hz)
        h_freq: Low-pass filter frequency (Hz)
        notch_freq: Notch filter frequency (Hz)
        verbose: Verbosity level

    Returns:
        Filtered Raw object
    """
    logger.info(f"Applying filters: HP={l_freq}Hz, LP={h_freq}Hz, Notch={notch_freq}Hz")

    # Apply band-pass filter
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    # Apply notch filter for line noise
    if notch_freq is not None:
        freqs = np.arange(notch_freq, min(h_freq or 100, raw.info['sfreq'] / 2), notch_freq)
        raw.notch_filter(freqs=freqs, verbose=verbose)

    return raw


def remove_artifacts(
    raw: mne.io.BaseRaw,
    method: str = 'ica',
    n_components: int = 20,
    random_state: int = 42,
    cutoff: float = 20,
    verbose: bool = False
) -> mne.io.BaseRaw:
    """
    Remove artifacts using ICA or ASR

    Args:
        raw: MNE Raw object
        method: Artifact removal method ('ica' or 'asr')
        n_components: Number of ICA components
        random_state: Random seed for ICA
        cutoff: ASR cutoff parameter
        verbose: Verbosity level

    Returns:
        Cleaned Raw object
    """
    if method == 'ica':
        logger.info(f"Removing artifacts using ICA with {n_components} components")

        # Fit ICA
        ica = ICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=800,
            verbose=verbose
        )

        # Filter data for ICA (1-40 Hz recommended)
        raw_filt = raw.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False)
        ica.fit(raw_filt)

        # Find EOG artifacts
        eog_indices = []
        if 'eog' in raw.get_channel_types():
            eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=verbose)
        else:
            # Use frontal channels as EOG proxy
            eog_indices, eog_scores = ica.find_bads_eog(
                raw,
                ch_name=['EEG001', 'EEG002'] if 'EEG001' in raw.ch_names else raw.ch_names[:2],
                verbose=verbose
            )

        # Find ECG artifacts
        ecg_indices = []
        if 'ecg' in raw.get_channel_types():
            ecg_indices, ecg_scores = ica.find_bads_ecg(raw, verbose=verbose)

        # Exclude artifact components
        ica.exclude = list(set(eog_indices + ecg_indices))
        logger.info(f"Excluding {len(ica.exclude)} artifact components")

        # Apply ICA
        ica.apply(raw)

    elif method == 'asr':
        logger.info(f"Removing artifacts using ASR with cutoff={cutoff}")

        # ASR implementation would go here
        # For now, using simple threshold-based cleaning
        data = raw.get_data()
        std_threshold = cutoff * np.median(np.std(data, axis=1))

        # Find and interpolate bad segments
        bad_segments = np.any(np.abs(data) > std_threshold, axis=0)
        if np.any(bad_segments):
            logger.info(f"Found {np.sum(bad_segments)} bad samples")
            # Simple interpolation (in practice, use more sophisticated methods)
            for ch_idx in range(data.shape[0]):
                data[ch_idx, bad_segments] = np.interp(
                    np.where(bad_segments)[0],
                    np.where(~bad_segments)[0],
                    data[ch_idx, ~bad_segments]
                )
            raw._data = data

    else:
        raise ValueError(f"Unknown artifact removal method: {method}")

    return raw


def re_reference(
    raw: mne.io.BaseRaw,
    ref_type: str = 'average',
    ref_channels: Optional[List[str]] = None,
    verbose: bool = False
) -> mne.io.BaseRaw:
    """
    Apply re-referencing to EEG data

    Args:
        raw: MNE Raw object
        ref_type: Reference type ('average', 'rest', 'channel')
        ref_channels: Reference channels for 'channel' type
        verbose: Verbosity level

    Returns:
        Re-referenced Raw object
    """
    logger.info(f"Applying {ref_type} reference")

    if ref_type == 'average':
        raw.set_eeg_reference('average', projection=False, verbose=verbose)

    elif ref_type == 'rest':
        # REST reference requires forward model
        raw.set_eeg_reference('REST', verbose=verbose)

    elif ref_type == 'channel':
        if ref_channels is None:
            raise ValueError("ref_channels must be specified for channel reference")
        raw.set_eeg_reference(ref_channels, verbose=verbose)

    else:
        raise ValueError(f"Unknown reference type: {ref_type}")

    return raw


def segment_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: Dict[str, int],
    tmin: float = -0.2,
    tmax: float = 0.8,
    baseline: tuple = (-0.2, 0),
    reject: Optional[Dict[str, float]] = None,
    verbose: bool = False
) -> mne.Epochs:
    """
    Segment continuous data into epochs

    Args:
        raw: MNE Raw object
        events: Event array
        event_id: Event ID dictionary
        tmin: Start time before event (seconds)
        tmax: End time after event (seconds)
        baseline: Baseline correction window
        reject: Rejection thresholds
        verbose: Verbosity level

    Returns:
        MNE Epochs object
    """
    logger.info(f"Creating epochs: tmin={tmin}, tmax={tmax}, baseline={baseline}")

    if reject is None:
        # Default rejection thresholds
        reject = dict(eeg=100e-6)  # 100 µV

    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        verbose=verbose
    )

    logger.info(f"Created {len(epochs)} epochs from {len(events)} events")
    logger.info(f"Dropped {len(events) - len(epochs)} bad epochs")

    return epochs


def preprocess_raw(
    raw: mne.io.BaseRaw,
    config: Dict[str, Any],
    remove_artifacts: bool = True,
    re_reference: bool = True,
    verbose: bool = False
) -> mne.io.BaseRaw:
    """
    Complete preprocessing pipeline for raw EEG data

    Args:
        raw: MNE Raw object
        config: Configuration dictionary
        remove_artifacts: Whether to remove artifacts
        re_reference: Whether to re-reference
        verbose: Verbosity level

    Returns:
        Preprocessed Raw object
    """
    # Apply filters
    raw = apply_filters(
        raw,
        l_freq=config.get('highpass_hz', 0.5),
        h_freq=config.get('lowpass_hz', 40.0),
        notch_freq=config.get('notch_hz'),
        verbose=verbose
    )

    # Remove artifacts
    if remove_artifacts:
        raw = remove_artifacts(
            raw,
            method='ica',
            n_components=20,
            verbose=verbose
        )

    # Re-reference
    if re_reference:
        raw = re_reference(
            raw,
            ref_type='average',
            verbose=verbose
        )

    return raw


class PreprocessingPipeline:
    """Complete preprocessing pipeline with batch processing support"""

    def __init__(self, config: Union[str, Path, Dict[str, Any]]):
        """
        Initialize preprocessing pipeline

        Args:
            config: Configuration file path or dictionary
        """
        if isinstance(config, (str, Path)):
            self.config = load_config(config)
        else:
            self.config = config

        logger.info("Initialized preprocessing pipeline")

    def process(
        self,
        raw: mne.io.BaseRaw,
        apply_filters: bool = True,
        remove_artifacts: bool = True,
        re_reference: bool = True,
        reference_type: str = 'average',
        interpolate_bads: bool = True,
        verbose: bool = False
    ) -> mne.io.BaseRaw:
        """
        Process single raw file

        Args:
            raw: MNE Raw object
            apply_filters: Whether to apply frequency filters
            remove_artifacts: Whether to remove artifacts
            re_reference: Whether to re-reference data
            reference_type: Type of reference to apply
            interpolate_bads: Whether to interpolate bad channels
            verbose: Verbosity level

        Returns:
            Processed Raw object
        """
        logger.info(f"Processing raw data with {len(raw.ch_names)} channels")

        # Interpolate bad channels
        if interpolate_bads and len(raw.info['bads']) > 0:
            logger.info(f"Interpolating {len(raw.info['bads'])} bad channels")
            raw.interpolate_bads(reset_bads=True, verbose=verbose)

        # Apply filters
        if apply_filters:
            raw = apply_filters(
                raw,
                l_freq=self.config.get('highpass_hz', 0.5),
                h_freq=self.config.get('lowpass_hz', 40.0),
                notch_freq=self.config.get('notch_hz'),
                verbose=verbose
            )

        # Remove artifacts
        if remove_artifacts:
            raw = remove_artifacts(
                raw,
                method='ica',
                n_components=min(20, len(raw.ch_names) - 1),
                verbose=verbose
            )

        # Re-reference
        if re_reference:
            raw = re_reference(
                raw,
                ref_type=reference_type,
                verbose=verbose
            )

        return raw

    def batch_process(
        self,
        subjects_data: Dict[str, mne.io.BaseRaw],
        output_dir: Optional[Path] = None,
        save_intermediate: bool = False,
        **kwargs
    ) -> Dict[str, mne.io.BaseRaw]:
        """
        Process multiple subjects

        Args:
            subjects_data: Dictionary of subject_id: Raw object
            output_dir: Directory to save processed data
            save_intermediate: Whether to save intermediate files
            **kwargs: Additional processing parameters

        Returns:
            Dictionary of processed data
        """
        processed_data = {}

        for sub_id, raw in subjects_data.items():
            logger.info(f"Processing subject {sub_id}")

            try:
                processed = self.process(raw, **kwargs)
                processed_data[sub_id] = processed

                if save_intermediate and output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    output_file = output_dir / f"{sub_id}_preprocessed_raw.fif"
                    processed.save(output_file, overwrite=True)
                    logger.info(f"Saved processed data to {output_file}")

            except Exception as e:
                logger.error(f"Failed to process {sub_id}: {e}")
                continue

        logger.info(f"Processed {len(processed_data)}/{len(subjects_data)} subjects")
        return processed_data

    def save_processed(
        self,
        raw: mne.io.BaseRaw,
        output_path: Path,
        save_format: str = 'fif'
    ) -> None:
        """
        Save processed data

        Args:
            raw: Processed Raw object
            output_path: Output file path
            save_format: File format ('fif' or 'edf')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if save_format == 'fif':
            raw.save(output_path, overwrite=True)
        elif save_format == 'edf':
            mne.export.export_raw(output_path, raw, fmt='edf', overwrite=True)
        else:
            raise ValueError(f"Unknown save format: {save_format}")

        logger.info(f"Saved processed data to {output_path}")


def main():
    """Main entry point for preprocessing"""
    ap = argparse.ArgumentParser(description="Preprocess EEG data")
    ap.add_argument('--config', default='configs/bands.yaml',
                   help='Configuration file')
    ap.add_argument('--input', type=str, required=True,
                   help='Input raw data file')
    ap.add_argument('--output', type=str,
                   help='Output processed data file')
    ap.add_argument('--no-filter', action='store_true',
                   help='Skip filtering')
    ap.add_argument('--no-ica', action='store_true',
                   help='Skip ICA artifact removal')
    ap.add_argument('--reference', choices=['average', 'rest', 'none'],
                   default='average', help='Reference type')
    args = ap.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load raw data
    logger.info(f"Loading raw data from {args.input}")
    raw = mne.io.read_raw_fif(args.input, preload=True)

    # Initialize pipeline
    pipeline = PreprocessingPipeline(config)

    # Process data
    processed = pipeline.process(
        raw,
        apply_filters=not args.no_filter,
        remove_artifacts=not args.no_ica,
        re_reference=(args.reference != 'none'),
        reference_type=args.reference
    )

    # Save processed data
    if args.output:
        pipeline.save_processed(processed, Path(args.output))

    logger.info("Preprocessing completed")


if __name__ == "__main__":
    main()
