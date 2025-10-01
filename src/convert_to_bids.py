"""
BIDS-EEG Conversion Module for EEG-MCI-Bench
Converts raw EEG data to BIDS format with strict compliance
"""
import argparse
import os
import json
import pathlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import mne
from mne_bids import BIDSPath, write_raw_bids, print_dir_tree, make_dataset_description
from mne_bids.utils import _write_json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_bids_structure(bids_root: Path) -> bool:
    """
    Validate BIDS directory structure

    Args:
        bids_root: Root directory of BIDS dataset

    Returns:
        bool: True if valid BIDS structure
    """
    if not bids_root.exists():
        return False

    # Check required files
    required_files = ['participants.tsv', 'dataset_description.json']
    for file in required_files:
        if not (bids_root / file).exists():
            logger.warning(f"Missing required file: {file}")
            return False

    # Check for at least one subject directory
    subject_dirs = list(bids_root.glob('sub-*'))
    if not subject_dirs:
        logger.warning("No subject directories found")
        return False

    return True


def create_dataset_description(
    name: str = "EEG-MCI-Bench",
    bids_version: str = "1.9.0",
    authors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create dataset description dictionary

    Args:
        name: Dataset name
        bids_version: BIDS specification version
        authors: List of authors

    Returns:
        dict: Dataset description
    """
    description = {
        "Name": name,
        "BIDSVersion": bids_version,
        "DatasetType": "raw",
        "Authors": authors or ["EEG-MCI-Bench Contributors"],
        "Acknowledgements": "MCI detection benchmark dataset",
        "License": "CC-BY-4.0",
        "ReferencesAndLinks": [
            "https://github.com/eeg-mci-bench/eeg-mci-benchmark"
        ],
        "DatasetDOI": "pending",
        "GeneratedBy": [{
            "Name": "EEG-MCI-Bench Pipeline",
            "Version": "1.0.0"
        }],
        "EthicsApprovals": ["IRB approved for cognitive assessment research"]
    }

    return description


def create_participants_json() -> Dict[str, Any]:
    """
    Create participants.json sidecar with MCI-specific fields

    Returns:
        dict: Participants JSON structure
    """
    participants_json = {
        "participant_id": {
            "Description": "Unique participant identifier"
        },
        "age": {
            "Description": "Age of participant at time of recording",
            "Units": "years"
        },
        "sex": {
            "Description": "Biological sex of participant",
            "Levels": {
                "M": "Male",
                "F": "Female"
            }
        },
        "group": {
            "Description": "Clinical group classification",
            "Levels": {
                "control": "Healthy control",
                "MCI": "Mild Cognitive Impairment"
            }
        },
        "mmse_score": {
            "Description": "Mini-Mental State Examination score",
            "Units": "score points (0-30)"
        },
        "moca_score": {
            "Description": "Montreal Cognitive Assessment score",
            "Units": "score points (0-30)"
        },
        "education_years": {
            "Description": "Years of formal education",
            "Units": "years"
        },
        "handedness": {
            "Description": "Dominant hand",
            "Levels": {
                "R": "Right",
                "L": "Left",
                "A": "Ambidextrous"
            }
        }
    }

    return participants_json


def convert_raw_to_bids(
    raw: mne.io.BaseRaw,
    subject_id: str,
    session: Optional[str] = None,
    task: str = "rest",
    bids_root: Path = Path("data/bids_raw"),
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = True
) -> BIDSPath:
    """
    Convert raw EEG data to BIDS format

    Args:
        raw: MNE Raw object
        subject_id: Subject identifier
        session: Session identifier (optional)
        task: Task name
        bids_root: Root directory for BIDS dataset
        metadata: Subject metadata dictionary
        overwrite: Whether to overwrite existing files

    Returns:
        BIDSPath: Path to created BIDS file
    """
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)

    # Create dataset description if not exists
    desc_file = bids_root / "dataset_description.json"
    if not desc_file.exists():
        desc = create_dataset_description()
        _write_json(desc_file, desc, overwrite=True)

    # Create participants.json if not exists
    participants_json_file = bids_root / "participants.json"
    if not participants_json_file.exists():
        _write_json(participants_json_file, create_participants_json(), overwrite=True)

    # Create BIDSPath
    bids_path = BIDSPath(
        subject=subject_id,
        session=session,
        task=task,
        datatype='eeg',
        root=bids_root
    )

    # Write raw data to BIDS
    write_raw_bids(
        raw,
        bids_path,
        overwrite=overwrite,
        format='BrainVision',
        verbose=False
    )

    # Update participants.tsv with metadata
    if metadata:
        update_participants_tsv(bids_root, subject_id, metadata)

    logger.info(f"Converted subject {subject_id} to BIDS format at {bids_path.fpath}")

    return bids_path


def update_participants_tsv(
    bids_root: Path,
    subject_id: str,
    metadata: Dict[str, Any]
) -> None:
    """
    Update participants.tsv with subject metadata

    Args:
        bids_root: BIDS root directory
        subject_id: Subject identifier
        metadata: Subject metadata dictionary
    """
    participants_file = bids_root / "participants.tsv"

    # Load existing participants.tsv or create new
    if participants_file.exists():
        df = pd.read_csv(participants_file, sep='\t')
    else:
        df = pd.DataFrame()

    # Add participant_id with sub- prefix
    metadata['participant_id'] = f'sub-{subject_id}'

    # Update or add row
    if 'participant_id' in df.columns and f'sub-{subject_id}' in df['participant_id'].values:
        # Update existing
        idx = df[df['participant_id'] == f'sub-{subject_id}'].index[0]
        for key, value in metadata.items():
            df.loc[idx, key] = value
    else:
        # Add new row
        df = pd.concat([df, pd.DataFrame([metadata])], ignore_index=True)

    # Save
    df.to_csv(participants_file, sep='\t', index=False)


def check_bids_compliance(bids_root: Path) -> Tuple[bool, List[str]]:
    """
    Check BIDS compliance of dataset

    Args:
        bids_root: Root directory of BIDS dataset

    Returns:
        tuple: (is_compliant, list of errors)
    """
    errors = []
    bids_root = Path(bids_root)

    # Check required files
    if not (bids_root / "dataset_description.json").exists():
        errors.append("Missing dataset_description.json")

    if not (bids_root / "participants.tsv").exists():
        errors.append("Missing participants.tsv")

    # Check subject directories
    subject_dirs = list(bids_root.glob('sub-*'))
    if not subject_dirs:
        errors.append("No subject directories found")

    # Check EEG data files
    for sub_dir in subject_dirs:
        eeg_files = list(sub_dir.rglob('*_eeg.*'))
        if not eeg_files:
            errors.append(f"No EEG files found for {sub_dir.name}")

    is_compliant = len(errors) == 0

    if not is_compliant:
        logger.warning(f"BIDS compliance check failed with {len(errors)} errors")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("BIDS compliance check passed")

    return is_compliant, errors


def batch_convert_to_bids(
    data_dir: Path,
    bids_root: Path,
    file_pattern: str = "*.edf",
    task: str = "rest"
) -> None:
    """
    Batch convert multiple EEG files to BIDS format

    Args:
        data_dir: Directory containing raw EEG files
        bids_root: Output BIDS directory
        file_pattern: Pattern for finding EEG files
        task: Task name for all files
    """
    data_dir = Path(data_dir)
    files = list(data_dir.glob(file_pattern))

    logger.info(f"Found {len(files)} files to convert")

    for idx, file in enumerate(files, 1):
        logger.info(f"Processing file {idx}/{len(files)}: {file.name}")

        try:
            # Extract subject ID from filename (customize as needed)
            subject_id = file.stem.split('_')[0]

            # Read raw data
            if file.suffix.lower() == '.edf':
                raw = mne.io.read_raw_edf(file, preload=False, verbose=False)
            elif file.suffix.lower() == '.bdf':
                raw = mne.io.read_raw_bdf(file, preload=False, verbose=False)
            else:
                logger.warning(f"Unsupported file format: {file.suffix}")
                continue

            # Convert to BIDS
            convert_raw_to_bids(
                raw=raw,
                subject_id=subject_id,
                task=task,
                bids_root=bids_root
            )

        except Exception as e:
            logger.error(f"Failed to convert {file.name}: {e}")
            continue

    logger.info("Batch conversion completed")


def main():
    """Main entry point for BIDS conversion"""
    ap = argparse.ArgumentParser(description="Convert EEG data to BIDS format")
    ap.add_argument('--input', type=str, help='Input data directory')
    ap.add_argument('--output', type=str, default='data/bids_raw',
                   help='Output BIDS directory')
    ap.add_argument('--validate', action='store_true',
                   help='Validate BIDS structure')
    ap.add_argument('--pattern', type=str, default='*.edf',
                   help='File pattern for batch conversion')
    ap.add_argument('--task', type=str, default='rest',
                   help='Task name for BIDS')
    args = ap.parse_args()

    bids_root = Path(args.output)

    if args.validate:
        if validate_bids_structure(bids_root):
            print_dir_tree(bids_root)
            is_compliant, errors = check_bids_compliance(bids_root)
            if is_compliant:
                logger.info("✓ BIDS validation passed")
            else:
                logger.error(f"✗ BIDS validation failed with {len(errors)} errors")
        else:
            logger.error("Invalid BIDS structure")
    elif args.input:
        batch_convert_to_bids(
            data_dir=Path(args.input),
            bids_root=bids_root,
            file_pattern=args.pattern,
            task=args.task
        )
    else:
        logger.info("Use --input for batch conversion or --validate to check BIDS compliance")


if __name__ == '__main__':
    main()
