"""
Test suite for BIDS conversion module following TDD principles.
Tests must pass before implementation is considered complete.
"""
import pytest
import numpy as np
import mne
from pathlib import Path
import tempfile
import shutil
from mne_bids import BIDSPath, read_raw_bids
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.convert_to_bids import (
    validate_bids_structure,
    convert_raw_to_bids,
    create_participants_json,
    create_dataset_description,
    check_bids_compliance
)

class TestBIDSConversion:
    """Test BIDS conversion functionality with strict validation"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create temporary directory for test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.bids_root = Path(self.temp_dir) / 'bids_raw'
        self.bids_root.mkdir(parents=True)
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_raw_data(self):
        """Create mock EEG data for testing"""
        sfreq = 256
        n_channels = 64
        n_times = sfreq * 60  # 60 seconds
        info = mne.create_info(
            ch_names=[f'EEG{i:03d}' for i in range(1, n_channels+1)],
            sfreq=sfreq,
            ch_types='eeg'
        )
        data = np.random.randn(n_channels, n_times) * 1e-6
        raw = mne.io.RawArray(data, info)
        raw.set_montage('standard_1020')
        return raw

    def test_validate_bids_structure(self):
        """Test BIDS directory structure validation"""
        # Test empty directory fails
        assert not validate_bids_structure(self.bids_root)

        # Create minimal BIDS structure
        (self.bids_root / 'participants.tsv').touch()
        (self.bids_root / 'dataset_description.json').touch()
        (self.bids_root / 'sub-001').mkdir()

        # Should now pass basic validation
        assert validate_bids_structure(self.bids_root)

    def test_create_dataset_description(self):
        """Test dataset description JSON creation"""
        desc = create_dataset_description(
            name="EEG-MCI-Bench",
            bids_version="1.9.0",
            authors=["Test Author"]
        )

        assert desc['Name'] == "EEG-MCI-Bench"
        assert desc['BIDSVersion'] == "1.9.0"
        assert 'Authors' in desc
        assert 'DatasetType' in desc
        assert desc['DatasetType'] == 'raw'

    def test_create_participants_json(self):
        """Test participants JSON sidecar creation"""
        participants_json = create_participants_json()

        assert 'participant_id' in participants_json
        assert 'age' in participants_json
        assert 'sex' in participants_json
        assert 'group' in participants_json

        # Check MCI-specific fields
        assert participants_json['group']['Description']
        assert participants_json['group']['Levels']['control'] == 'Healthy control'
        assert participants_json['group']['Levels']['MCI'] == 'Mild Cognitive Impairment'

    def test_convert_single_subject(self):
        """Test converting single subject to BIDS format"""
        raw = self.create_mock_raw_data()

        subject_id = '001'
        session = '01'
        task = 'rest'

        bids_path = convert_raw_to_bids(
            raw=raw,
            subject_id=subject_id,
            session=session,
            task=task,
            bids_root=self.bids_root
        )

        # Verify file was created
        assert bids_path.fpath.exists()

        # Verify can be read back
        raw_read = read_raw_bids(bids_path)
        assert raw_read.info['sfreq'] == raw.info['sfreq']
        assert len(raw_read.ch_names) == len(raw.ch_names)

    def test_subject_metadata(self):
        """Test subject metadata is correctly saved"""
        raw = self.create_mock_raw_data()

        metadata = {
            'age': 65,
            'sex': 'M',
            'group': 'MCI',
            'mmse_score': 24,
            'education_years': 12
        }

        bids_path = convert_raw_to_bids(
            raw=raw,
            subject_id='002',
            session='01',
            task='rest',
            bids_root=self.bids_root,
            metadata=metadata
        )

        # Read participants.tsv and verify metadata
        participants_file = self.bids_root / 'participants.tsv'
        assert participants_file.exists()

        import pandas as pd
        df = pd.read_csv(participants_file, sep='\\t')
        subject_row = df[df['participant_id'] == 'sub-002']

        assert not subject_row.empty
        assert subject_row['age'].values[0] == 65
        assert subject_row['sex'].values[0] == 'M'
        assert subject_row['group'].values[0] == 'MCI'

    def test_bids_compliance_check(self):
        """Test BIDS compliance checking"""
        raw = self.create_mock_raw_data()

        # Convert multiple subjects
        for i in range(1, 4):
            convert_raw_to_bids(
                raw=raw,
                subject_id=f'{i:03d}',
                session='01',
                task='rest',
                bids_root=self.bids_root
            )

        # Check compliance
        is_compliant, errors = check_bids_compliance(self.bids_root)

        assert is_compliant, f"BIDS compliance failed: {errors}"
        assert len(errors) == 0

    def test_multiple_sessions(self):
        """Test handling multiple sessions per subject"""
        raw = self.create_mock_raw_data()

        subject_id = '003'
        sessions = ['01', '02', '03']

        for session in sessions:
            bids_path = convert_raw_to_bids(
                raw=raw,
                subject_id=subject_id,
                session=session,
                task='rest',
                bids_root=self.bids_root
            )
            assert bids_path.fpath.exists()

        # Verify all sessions exist
        subject_dir = self.bids_root / f'sub-{subject_id}'
        session_dirs = list(subject_dir.glob('ses-*'))
        assert len(session_dirs) == 3

    def test_event_markers(self):
        """Test event/stimulus markers are preserved"""
        raw = self.create_mock_raw_data()

        # Add events
        events = np.array([
            [256, 0, 1],   # 1 second: stimulus
            [512, 0, 2],   # 2 seconds: response
            [768, 0, 1],   # 3 seconds: stimulus
        ])

        annotations = mne.annotations_from_events(
            events, sfreq=raw.info['sfreq'],
            event_desc={1: 'stimulus', 2: 'response'}
        )
        raw.set_annotations(annotations)

        bids_path = convert_raw_to_bids(
            raw=raw,
            subject_id='004',
            session='01',
            task='cognitive',
            bids_root=self.bids_root
        )

        # Read back and verify events
        raw_read = read_raw_bids(bids_path)
        assert len(raw_read.annotations) == 3
        assert 'stimulus' in raw_read.annotations.description
        assert 'response' in raw_read.annotations.description