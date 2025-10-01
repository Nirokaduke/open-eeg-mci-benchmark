#!/usr/bin/env python
"""
Preprocessing pipeline for resting-state EEG data
"""
import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# MNE imports
import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_subject(
    bids_path: BIDSPath,
    config: Dict,
    output_dir: Path,
    use_ica: bool = True,
    use_asr: bool = False
) -> mne.io.Raw:
    """
    Preprocess a single subject's EEG data

    Parameters
    ----------
    bids_path : BIDSPath
        Path to subject's BIDS data
    config : dict
        Configuration parameters
    output_dir : Path
        Directory to save preprocessed data
    use_ica : bool
        Whether to apply ICA for artifact removal
    use_asr : bool
        Whether to apply ASR (Artifact Subspace Reconstruction)

    Returns
    -------
    raw : mne.io.Raw
        Preprocessed raw data
    """
    # Read raw data
    raw = read_raw_bids(bids_path, verbose=False)

    # Get sampling rate
    sfreq = raw.info['sfreq']
    print(f"  Original sampling rate: {sfreq} Hz")

    # Resample if needed
    if 'sampling_rate' in config and config['sampling_rate'] != sfreq:
        print(f"  Resampling to {config['sampling_rate']} Hz")
        raw.resample(config['sampling_rate'])

    # Apply notch filter if specified
    if config.get('notch_hz'):
        print(f"  Applying notch filter at {config['notch_hz']} Hz")
        raw.notch_filter(config['notch_hz'])

    # Apply bandpass filter
    l_freq = config.get('highpass_hz', 0.5)
    h_freq = config.get('lowpass_hz', 40.0)
    print(f"  Applying bandpass filter: {l_freq}-{h_freq} Hz")
    raw.filter(l_freq, h_freq, fir_design='firwin')

    # Re-reference to average
    print("  Re-referencing to average")
    raw.set_eeg_reference('average', projection=False)

    # ICA for artifact removal
    if use_ica:
        print("  Applying ICA for artifact removal")
        n_eeg = len(mne.pick_types(raw.info, meg=False, eeg=True))
        n_components = min(20, n_eeg - 1)  # Use at most n_channels - 1
        ica = ICA(n_components=n_components, random_state=42, max_iter=800)
        ica.fit(raw, picks='eeg')

        # Automatic component selection for EOG/ECG artifacts
        try:
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=None)
        except:
            eog_indices = []
            print("    No EOG channels found for artifact detection")

        try:
            ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name=None)
        except:
            ecg_indices = []
            print("    No ECG channels found for artifact detection")

        # Exclude bad components
        ica.exclude = list(set(eog_indices[:2] + ecg_indices[:1]))  # Conservative selection
        if ica.exclude:
            print(f"    Excluding ICA components: {ica.exclude}")
            raw = ica.apply(raw)
        else:
            print("    No artifacts automatically detected, keeping all components")

    # ASR (simplified version - in practice would use pyprep or similar)
    if use_asr:
        print("  ASR not implemented in this version")

    # Save preprocessed data
    output_file = output_dir / f"{bids_path.subject}_preprocessed_raw.fif"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_file, overwrite=True)
    print(f"  Saved to {output_file}")

    return raw

def main():
    parser = argparse.ArgumentParser(description='Preprocess EEG data')
    parser.add_argument('--bids_root', type=str, default='data/bids_raw/ds004504',
                       help='Path to BIDS dataset root')
    parser.add_argument('--config', type=str, default='configs/bands.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='data/derivatives/clean',
                       help='Output directory for preprocessed data')
    parser.add_argument('--subject', type=str, default=None,
                       help='Process specific subject (e.g., "001")')
    parser.add_argument('--use_ica', action='store_true', default=True,
                       help='Apply ICA for artifact removal')
    parser.add_argument('--use_asr', action='store_true', default=False,
                       help='Apply ASR for artifact removal')
    parser.add_argument('--report_file', type=str, default='reports/preprocessing.md',
                       help='Path to save preprocessing report')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup paths
    bids_root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize report
    report_lines = [
        "# Preprocessing Report",
        f"\n## Configuration",
        f"- BIDS root: {bids_root}",
        f"- Output directory: {output_dir}",
        f"- Config file: {args.config}",
        f"- ICA: {'Enabled' if args.use_ica else 'Disabled'}",
        f"- ASR: {'Enabled' if args.use_asr else 'Disabled'}",
        f"\n## Processing Parameters",
        f"- Sampling rate: {config.get('sampling_rate', 'Original')} Hz",
        f"- Highpass: {config.get('highpass_hz', 0.5)} Hz",
        f"- Lowpass: {config.get('lowpass_hz', 40.0)} Hz",
        f"- Notch: {config.get('notch_hz', 'None')} Hz",
        f"- Reference: Average",
        f"\n## Subjects Processed\n"
    ]

    # Find subjects to process
    if args.subject:
        subjects = [args.subject]
    else:
        # Find all subjects in BIDS dataset
        subject_dirs = [d for d in bids_root.iterdir()
                       if d.is_dir() and d.name.startswith('sub-')]
        subjects = [d.name.replace('sub-', '') for d in subject_dirs]

        if not subjects:
            print(f"No subjects found in {bids_root}")
            print("Looking for participants.tsv...")
            participants_file = bids_root / 'participants.tsv'
            if participants_file.exists():
                participants = pd.read_csv(participants_file, sep='\t')
                print(f"Found {len(participants)} participants in participants.tsv")
                print(participants.head())
            else:
                print("No participants.tsv found")
            sys.exit(1)

    print(f"\nProcessing {len(subjects)} subject(s)")

    # Process each subject
    processed_count = 0
    for subject in subjects:
        print(f"\nProcessing subject {subject}...")

        try:
            # Find EEG file for this subject
            subject_dir = bids_root / f"sub-{subject}" / "eeg"
            eeg_files = []
            if subject_dir.exists():
                eeg_files = list(subject_dir.glob("*.edf")) + list(subject_dir.glob("*.bdf")) + list(subject_dir.glob("*.set"))

            if not eeg_files:
                print(f"  Warning: No EEG data found for subject {subject}")
                report_lines.append(f"- sub-{subject}: Not found")
                continue

            # Use the first EEG file found
            eeg_file = eeg_files[0]
            print(f"  Found EEG file: {eeg_file.name}")

            # Parse filename to get task (if any)
            import re
            match = re.search(r'task-([^_]+)', eeg_file.name)
            task = match.group(1) if match else None

            # Create BIDS path
            if task:
                bids_path = BIDSPath(
                    subject=subject,
                    task=task,
                    datatype='eeg',
                    root=bids_root
                )
            else:
                # For datasets without task field
                bids_path = BIDSPath(
                    subject=subject,
                    datatype='eeg',
                    root=bids_root,
                    suffix='eeg'
                )

            # Preprocess
            raw = preprocess_subject(
                bids_path=bids_path,
                config=config,
                output_dir=output_dir,
                use_ica=args.use_ica,
                use_asr=args.use_asr
            )

            processed_count += 1
            report_lines.append(f"- sub-{subject}: Successfully preprocessed ({len(raw.ch_names)} channels, {raw.n_times/raw.info['sfreq']:.1f}s)")

        except Exception as e:
            print(f"  Error processing subject {subject}: {str(e)}")
            report_lines.append(f"- sub-{subject}: Error - {str(e)}")

    # Add summary to report
    report_lines.extend([
        f"\n## Summary",
        f"- Total subjects: {len(subjects)}",
        f"- Successfully processed: {processed_count}",
        f"- Failed: {len(subjects) - processed_count}"
    ])

    # Save report
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"Processed: {processed_count}/{len(subjects)} subjects")
    print(f"Report saved to: {report_path}")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()