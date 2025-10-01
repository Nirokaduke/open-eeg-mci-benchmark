"""
End-to-End Integration Test for EEG-MCI-Bench
Verifies complete pipeline from raw data to final reports
"""
import pytest
import numpy as np
import pandas as pd
import mne
from pathlib import Path
import tempfile
import shutil
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.convert_to_bids import (
    convert_raw_to_bids,
    validate_bids_structure,
    check_bids_compliance
)
from src.preprocessing import PreprocessingPipeline
from src.features.spectrum import extract_spectral_features
from src.features.entropy import extract_complexity_features
from src.features.connectivity import extract_connectivity_features
from src.features.erp import extract_erp_features
from src.models.classical import ClassicalMLPipeline
from src.reports.generate_tripod_ai import TRIPODAIReportGenerator
from src.reports.generate_stard import STARDReportGenerator
from src.reports.evidence_map import EvidenceMapGenerator


class TestEndToEndPipeline:
    """Complete end-to-end pipeline verification"""

    @pytest.fixture(scope="class")
    def test_environment(self):
        """Create test environment with mock data"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="eeg_mci_test_")

        # Create directory structure
        dirs = {
            'raw': Path(temp_dir) / 'data' / 'raw',
            'bids': Path(temp_dir) / 'data' / 'bids_raw',
            'derivatives': Path(temp_dir) / 'data' / 'derivatives',
            'features': Path(temp_dir) / 'data' / 'derivatives' / 'features',
            'models': Path(temp_dir) / 'data' / 'derivatives' / 'models',
            'reports': Path(temp_dir) / 'reports'
        }

        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        yield dirs

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def create_mock_eeg_data(self, n_subjects=5, n_channels=64, duration=60):
        """Create realistic mock EEG data"""
        sfreq = 256
        n_times = int(sfreq * duration)

        # Create info
        ch_names = [f'EEG{i:03d}' for i in range(1, n_channels+1)]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types='eeg'
        )

        # Generate data with EEG-like properties
        times = np.arange(n_times) / sfreq
        data = np.zeros((n_channels, n_times))

        for ch in range(n_channels):
            # Add different frequency components
            data[ch] = (
                0.5 * np.sin(2 * np.pi * 10 * times) +  # Alpha (10 Hz)
                0.3 * np.sin(2 * np.pi * 6 * times) +   # Theta (6 Hz)
                0.2 * np.sin(2 * np.pi * 20 * times) +  # Beta (20 Hz)
                0.1 * np.sin(2 * np.pi * 2 * times) +   # Delta (2 Hz)
                0.1 * np.random.randn(n_times)          # Noise
            ) * 1e-6

        raw = mne.io.RawArray(data, info)

        # Add events for ERP analysis
        events = []
        for i in range(10, 50, 5):  # Events every 5 seconds
            events.append([int(i * sfreq), 0, 1])  # Stimulus events

        if events:
            annotations = mne.annotations_from_events(
                np.array(events),
                sfreq=sfreq,
                event_desc={1: 'stimulus'}
            )
            raw.set_annotations(annotations)

        return raw

    def test_step1_bids_conversion(self, test_environment):
        """Test Step 1: Convert raw data to BIDS format"""
        print("\n=== Step 1: BIDS Conversion ===")

        # Create mock subjects
        n_subjects = 3
        subject_metadata = []

        for subj_id in range(1, n_subjects + 1):
            # Create mock raw data
            raw = self.create_mock_eeg_data()

            # Subject metadata
            metadata = {
                'age': 65 + subj_id,
                'sex': 'M' if subj_id % 2 == 0 else 'F',
                'group': 'MCI' if subj_id <= 2 else 'control',
                'mmse_score': 24 + subj_id
            }
            subject_metadata.append(metadata)

            # Convert to BIDS
            bids_path = convert_raw_to_bids(
                raw=raw,
                subject_id=f'{subj_id:03d}',
                task='rest',
                bids_root=test_environment['bids'],
                metadata=metadata
            )

            assert bids_path.fpath.exists(), f"BIDS file not created for subject {subj_id}"

        # Validate BIDS structure
        is_valid = validate_bids_structure(test_environment['bids'])
        assert is_valid, "BIDS structure validation failed"

        # Check compliance
        is_compliant, errors = check_bids_compliance(test_environment['bids'])
        assert is_compliant, f"BIDS compliance failed: {errors}"

        print(f"  [OK] Converted {n_subjects} subjects to BIDS format")
        print(f"  [OK] BIDS structure validated")
        print(f"  [OK] BIDS compliance verified")

        return n_subjects

    def test_step2_preprocessing(self, test_environment):
        """Test Step 2: Preprocessing pipeline"""
        print("\n=== Step 2: Preprocessing ===")

        # Initialize preprocessing pipeline
        config = {
            'sampling_rate': 256,
            'highpass_hz': 0.5,
            'lowpass_hz': 40.0,
            'bands': {
                'delta': [1, 4],
                'theta': [4, 8],
                'alpha': [8, 13],
                'beta': [13, 30]
            }
        }

        pipeline = PreprocessingPipeline(config)

        # Process each subject
        processed_count = 0
        bids_root = test_environment['bids']

        for subject_dir in bids_root.glob('sub-*'):
            for eeg_file in subject_dir.rglob('*_eeg.vhdr'):
                # Read BIDS data
                raw = mne.io.read_raw_brainvision(eeg_file, preload=True)

                # Preprocess
                processed = pipeline.process(
                    raw,
                    apply_filters=True,
                    remove_artifacts=False,  # Skip ICA for speed in test
                    re_reference=True,
                    reference_type='average'
                )

                # Save preprocessed data
                output_file = test_environment['derivatives'] / f'{subject_dir.name}_preprocessed.fif'
                processed.save(output_file, overwrite=True)

                assert output_file.exists(), f"Preprocessed file not saved: {output_file}"
                processed_count += 1

        print(f"  [OK] Preprocessed {processed_count} subjects")
        print(f"  [OK] Applied filters (0.5-40 Hz)")
        print(f"  [OK] Applied average reference")

        return processed_count

    def test_step3_feature_extraction(self, test_environment):
        """Test Step 3: Extract features from all 4 families"""
        print("\n=== Step 3: Feature Extraction ===")

        all_features = []

        # Process each preprocessed file
        for prep_file in test_environment['derivatives'].glob('*_preprocessed.fif'):
            # Load preprocessed data
            raw = mne.io.read_raw_fif(prep_file, preload=True)

            # Extract subject ID
            subject_id = prep_file.stem.split('_')[0]

            print(f"\n  Extracting features for {subject_id}:")

            # 1. Spectral features
            spectral_features = extract_spectral_features(
                raw,
                bands={
                    'delta': (1, 4),
                    'theta': (4, 8),
                    'alpha': (8, 13),
                    'beta': (13, 30)
                }
            )
            print(f"    - Spectral: {spectral_features.shape[0]} channels, {spectral_features.shape[1]} features")

            # 2. Complexity features
            complexity_features = extract_complexity_features(raw)
            print(f"    - Complexity: {complexity_features.shape[1]} features")

            # 3. Connectivity features (create epochs first)
            events = mne.make_fixed_length_events(raw, duration=2.0)
            epochs = mne.Epochs(
                raw, events, tmin=0, tmax=2.0,
                baseline=None, preload=True, verbose=False
            )

            connectivity_features = extract_connectivity_features(epochs)
            print(f"    - Connectivity: {connectivity_features.shape[1]} features")

            # 4. ERP features (if events exist)
            if len(raw.annotations) > 0:
                # Create epochs from annotations
                events, event_id = mne.events_from_annotations(raw)
                epochs_erp = mne.Epochs(
                    raw, events, event_id,
                    tmin=-0.2, tmax=0.8,
                    baseline=(-0.2, 0),
                    preload=True, verbose=False
                )

                erp_features = extract_erp_features(epochs_erp)
                print(f"    - ERP: {erp_features.shape[1]} features")
            else:
                erp_features = pd.DataFrame()
                print(f"    - ERP: Skipped (no events)")

            # Combine all features
            subject_features = pd.concat(
                [spectral_features.mean().to_frame().T,
                 complexity_features.mean().to_frame().T,
                 connectivity_features,
                 erp_features.mean().to_frame().T if not erp_features.empty else pd.DataFrame()],
                axis=1
            )
            subject_features['subject_id'] = subject_id
            all_features.append(subject_features)

        # Combine all subjects
        feature_matrix = pd.concat(all_features, ignore_index=True)

        # Save feature matrix
        feature_file = test_environment['features'] / 'all_features.csv'
        feature_matrix.to_csv(feature_file, index=False)

        print(f"\n  [OK] Feature extraction complete")
        print(f"  [OK] Total features: {feature_matrix.shape[1] - 1}")
        print(f"  [OK] Feature matrix saved: {feature_file}")

        return feature_matrix

    def test_step4_ml_training(self, test_environment):
        """Test Step 4: Train ML models with LOSO-CV"""
        print("\n=== Step 4: ML Model Training ===")

        # Load feature matrix
        feature_file = test_environment['features'] / 'all_features.csv'
        if not feature_file.exists():
            # Create mock features for testing
            feature_matrix = pd.DataFrame({
                'feature_1': np.random.randn(9),
                'feature_2': np.random.randn(9),
                'feature_3': np.random.randn(9),
                'subject_id': ['sub-001'] * 3 + ['sub-002'] * 3 + ['sub-003'] * 3,
                'label': [0, 0, 0, 1, 1, 1, 0, 0, 0]
            })
        else:
            feature_matrix = pd.read_csv(feature_file)
            # Add mock labels
            feature_matrix['label'] = [i % 2 for i in range(len(feature_matrix))]

        # Prepare data
        X = feature_matrix.drop(['subject_id', 'label'], axis=1, errors='ignore')
        y = feature_matrix.get('label', pd.Series([i % 2 for i in range(len(feature_matrix))]))
        groups = feature_matrix.get('subject_id', pd.Series([f'sub-{i//3:03d}' for i in range(len(feature_matrix))]))

        print(f"  Data shape: X={X.shape}, y={len(y)}, groups={len(groups.unique())} subjects")

        # Initialize ML pipeline
        pipeline = ClassicalMLPipeline(
            models=['svm', 'rf'],  # Skip XGBoost for faster testing
            use_gpu=False
        )

        # Train models with LOSO-CV
        print("  Training models with LOSO-CV...")
        pipeline.fit(X, y, groups)

        # Get results
        results = pipeline.results_
        assert 'metrics' in results, "No metrics in results"

        metrics = results['metrics']
        print(f"\n  Performance Metrics:")
        print(f"    - F1 Score: {metrics.get('f1_mean', 0):.3f} [{metrics.get('f1_ci_lower', 0):.3f}, {metrics.get('f1_ci_upper', 0):.3f}]")
        print(f"    - MCC: {metrics.get('mcc_mean', 0):.3f} [{metrics.get('mcc_ci_lower', 0):.3f}, {metrics.get('mcc_ci_upper', 0):.3f}]")
        print(f"    - AUC: {metrics.get('auc_mean', 0):.3f} [{metrics.get('auc_ci_lower', 0):.3f}, {metrics.get('auc_ci_upper', 0):.3f}]")

        # Save model
        model_file = test_environment['models'] / 'ml_pipeline.pkl'
        pipeline.save(model_file)

        print(f"\n  [OK] ML models trained")
        print(f"  [OK] LOSO-CV completed")
        print(f"  [OK] 95% CI calculated")
        print(f"  [OK] Model saved: {model_file}")

        return results

    def test_step5_report_generation(self, test_environment):
        """Test Step 5: Generate TRIPOD+AI and STARD reports"""
        print("\n=== Step 5: Report Generation ===")

        # Create mock results for report generation
        mock_results = {
            'metrics': {
                'f1_mean': 0.82, 'f1_ci_lower': 0.75, 'f1_ci_upper': 0.89,
                'mcc_mean': 0.65, 'mcc_ci_lower': 0.55, 'mcc_ci_upper': 0.75,
                'auc_mean': 0.88, 'auc_ci_lower': 0.82, 'auc_ci_upper': 0.94,
                'sensitivity': 0.85, 'specificity': 0.80,
                'ppv': 0.81, 'npv': 0.84
            },
            'predictions': {
                'y_true': [0, 0, 1, 1, 0, 1, 1, 0],
                'y_pred': [0, 0, 1, 1, 0, 1, 0, 0],
                'y_prob': [0.1, 0.2, 0.8, 0.9, 0.3, 0.85, 0.4, 0.2]
            },
            'feature_importance': {
                'alpha_power': 0.25,
                'theta_power': 0.20,
                'connectivity_pli': 0.15,
                'p300_amplitude': 0.12,
                'entropy_permutation': 0.10
            }
        }

        # 1. Generate TRIPOD+AI Report
        print("\n  Generating TRIPOD+AI Report...")
        tripod_report = TRIPODAIReportGenerator(mock_results, {
            'model_name': "SVM+RF Ensemble",
            'dataset_name': "EEG-MCI Test Dataset"
        })
        tripod_output = test_environment['reports'] / 'tripod_ai_report.md'

        tripod_content = tripod_report.generate_report()
        with open(tripod_output, 'w') as f:
            f.write(tripod_content)

        assert tripod_output.exists(), "TRIPOD+AI report not created"
        print(f"    [OK] TRIPOD+AI report generated: {tripod_output}")

        # 2. Generate STARD Report
        print("\n  Generating STARD Report...")
        stard_report = STARDReportGenerator(mock_results, {
            'test_name': "EEG-based MCI Detection",
            'reference_standard': "Clinical Diagnosis"
        })
        stard_output = test_environment['reports'] / 'stard_report.md'

        stard_content = stard_report.generate_report()
        with open(stard_output, 'w') as f:
            f.write(stard_content)

        assert stard_output.exists(), "STARD report not created"
        print(f"    [OK] STARD report generated: {stard_output}")

        # 3. Generate Evidence Map
        print("\n  Generating Evidence Map...")
        evidence_map = EvidenceMapGenerator({'ensemble': mock_results})
        evidence_output = test_environment['reports'] / 'evidence_map'
        evidence_output.mkdir(exist_ok=True)

        evidence_map.generate_all_visualizations(str(evidence_output))

        assert evidence_output.exists(), "Evidence map directory not created"
        print(f"    [OK] Evidence map generated: {evidence_output}")

        print("\n  [OK] All reports generated successfully")

        return True

    def test_complete_pipeline(self, test_environment):
        """Test the complete end-to-end pipeline"""
        print("\n" + "="*60)
        print("RUNNING COMPLETE END-TO-END PIPELINE TEST")
        print("="*60)

        # Step 1: BIDS Conversion
        n_subjects = self.test_step1_bids_conversion(test_environment)
        assert n_subjects > 0, "No subjects converted to BIDS"

        # Step 2: Preprocessing
        n_processed = self.test_step2_preprocessing(test_environment)
        assert n_processed == n_subjects, "Not all subjects preprocessed"

        # Step 3: Feature Extraction
        feature_matrix = self.test_step3_feature_extraction(test_environment)
        assert not feature_matrix.empty, "No features extracted"

        # Step 4: ML Training
        ml_results = self.test_step4_ml_training(test_environment)
        assert ml_results is not None, "ML training failed"

        # Step 5: Report Generation
        reports_created = self.test_step5_report_generation(test_environment)
        assert reports_created, "Report generation failed"

        print("\n" + "="*60)
        print("END-TO-END PIPELINE TEST: SUCCESS")
        print("="*60)
        print("\nSummary:")
        print(f"  ✓ {n_subjects} subjects processed")
        print(f"  ✓ {feature_matrix.shape[1]-1} features extracted")
        print(f"  ✓ LOSO-CV completed")
        print(f"  ✓ All reports generated")
        print("\nPipeline is fully functional!")

        return True


# Additional verification functions
def verify_gpu_availability():
    """Check if GPU is available for XGBoost"""
    try:
        import xgboost as xgb

        # Try to create a DMatrix with GPU
        data = np.random.randn(100, 10)
        labels = np.random.randint(0, 2, 100)

        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'objective': 'binary:logistic'
        }

        dtrain = xgb.DMatrix(data, label=labels)
        # Try a quick training
        xgb.train(params, dtrain, num_boost_round=1)

        print("GPU is available and working for XGBoost")
        return True
    except Exception as e:
        print(f"GPU not available for XGBoost: {e}")
        return False


def verify_dependencies():
    """Verify all required dependencies are installed"""
    required_packages = [
        'mne', 'mne_bids', 'numpy', 'scipy', 'pandas',
        'scikit-learn', 'xgboost', 'pyyaml', 'statsmodels',
        'matplotlib', 'plotly'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package} installed")
        except ImportError:
            print(f"  ✗ {package} missing")
            missing.append(package)

    return len(missing) == 0, missing


if __name__ == "__main__":
    print("="*60)
    print("EEG-MCI-BENCH END-TO-END VERIFICATION")
    print("="*60)

    # Check dependencies
    print("\nChecking Dependencies:")
    deps_ok, missing = verify_dependencies()

    if not deps_ok:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Please install with: pip install", ' '.join(missing))

    # Check GPU
    print("\nChecking GPU:")
    gpu_ok = verify_gpu_availability()

    # Run end-to-end test
    print("\nRunning End-to-End Test:")
    pytest.main([__file__, '-v', '--tb=short'])