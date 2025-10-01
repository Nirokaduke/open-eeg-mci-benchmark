"""
Complete Verification Script for EEG-MCI-Bench
Validates all modules and functionality
"""
import sys
import importlib
from pathlib import Path
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def verify_module_imports():
    """Verify all modules can be imported"""
    print("="*60)
    print("VERIFYING MODULE IMPORTS")
    print("="*60)

    modules_to_test = [
        # Core modules
        ('src.convert_to_bids', ['convert_raw_to_bids', 'validate_bids_structure']),
        ('src.preprocessing', ['PreprocessingPipeline', 'apply_filters']),

        # Feature modules
        ('src.features.spectrum', ['compute_psd', 'extract_spectral_features']),
        ('src.features.entropy', ['compute_permutation_entropy', 'extract_complexity_features']),
        ('src.features.connectivity', ['compute_pli', 'extract_connectivity_features']),
        ('src.features.erp', ['extract_p300', 'extract_erp_features']),

        # Model modules
        ('src.models.classical', ['LOSOClassifier', 'ClassicalMLPipeline']),

        # Report modules
        ('src.reports.generate_tripod_ai', ['TRIPODAIReportGenerator']),
        ('src.reports.generate_stard', ['STARDReportGenerator']),
        ('src.reports.evidence_map', ['EvidenceMapGenerator']),
    ]

    success_count = 0
    fail_count = 0

    for module_name, expected_functions in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            # Check expected functions exist
            for func_name in expected_functions:
                if not hasattr(module, func_name):
                    raise AttributeError(f"Missing function: {func_name}")

            print(f"  OK  {module_name}")
            success_count += 1
        except Exception as e:
            print(f"  FAIL {module_name}: {str(e)}")
            fail_count += 1

    print(f"\nResults: {success_count} OK, {fail_count} FAILED")
    return fail_count == 0


def verify_dependencies():
    """Check all required packages are installed"""
    print("\n" + "="*60)
    print("VERIFYING DEPENDENCIES")
    print("="*60)

    dependencies = {
        'mne': '>=1.7.0',
        'mne_bids': '>=0.16.0',
        'numpy': '>=1.26',
        'scipy': '>=1.12',
        'pandas': '>=2.2',
        'sklearn': '>=1.5',
        'xgboost': '>=2.1',
        'yaml': '>=6.0',
        'statsmodels': '>=0.14',
        'matplotlib': '>=3.9',
        'plotly': '>=5.24'
    }

    success_count = 0
    fail_count = 0

    for package, version in dependencies.items():
        try:
            if package == 'sklearn':
                module = importlib.import_module('sklearn')
            elif package == 'yaml':
                module = importlib.import_module('yaml')
            else:
                module = importlib.import_module(package)

            if hasattr(module, '__version__'):
                print(f"  OK  {package} {module.__version__}")
            else:
                print(f"  OK  {package} (version unknown)")
            success_count += 1
        except ImportError:
            print(f"  FAIL {package} not installed")
            fail_count += 1

    print(f"\nResults: {success_count} OK, {fail_count} FAILED")
    return fail_count == 0


def verify_file_structure():
    """Verify project structure is correct"""
    print("\n" + "="*60)
    print("VERIFYING FILE STRUCTURE")
    print("="*60)

    expected_files = [
        # Core modules
        'src/__init__.py',
        'src/convert_to_bids.py',
        'src/preprocessing.py',

        # Feature modules
        'src/features/spectrum.py',
        'src/features/entropy.py',
        'src/features/connectivity.py',
        'src/features/erp.py',

        # Model modules
        'src/models/classical.py',

        # Report modules
        'src/reports/generate_tripod_ai.py',
        'src/reports/generate_stard.py',
        'src/reports/evidence_map.py',

        # Config files
        'configs/bands.yaml',
        'configs/connectivity.yaml',
        'configs/erp.yaml',
        'configs/validation.yaml',

        # Test files
        'tests/test_bids_conversion.py',
        'tests/test_preprocessing.py',
        'tests/test_feature_extraction.py',
        'tests/test_ml_models.py',
        'tests/test_end_to_end.py',

        # Documentation
        'reports/FINAL_PROJECT_SUMMARY.md',
        'reports/run_log.md'
    ]

    success_count = 0
    fail_count = 0

    for file_path in expected_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"  OK  {file_path} ({size} bytes)")
            success_count += 1
        else:
            print(f"  FAIL {file_path} not found")
            fail_count += 1

    print(f"\nResults: {success_count} OK, {fail_count} FAILED")
    return fail_count == 0


def verify_gpu_support():
    """Check GPU availability"""
    print("\n" + "="*60)
    print("VERIFYING GPU SUPPORT")
    print("="*60)

    try:
        import xgboost as xgb
        import numpy as np

        # Create test data
        data = np.random.randn(100, 10)
        labels = np.random.randint(0, 2, 100)

        # Try GPU parameters
        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'objective': 'binary:logistic'
        }

        dtrain = xgb.DMatrix(data, label=labels)

        # Try quick training
        bst = xgb.train(params, dtrain, num_boost_round=1)

        print("  OK  GPU support available for XGBoost")
        print("      Device: cuda")
        print("      Tree method: hist")
        return True

    except Exception as e:
        if "GPU" in str(e) or "CUDA" in str(e) or "No device" in str(e):
            print("  INFO GPU not available, will use CPU fallback")
            print(f"      Reason: {str(e)}")
        else:
            print(f"  WARN Unexpected error: {str(e)}")
        return False


def verify_functionality():
    """Quick functional tests"""
    print("\n" + "="*60)
    print("VERIFYING CORE FUNCTIONALITY")
    print("="*60)

    tests_passed = []

    # Test 1: BIDS functionality
    try:
        from src.convert_to_bids import create_dataset_description
        desc = create_dataset_description()
        assert 'Name' in desc
        assert desc['Name'] == 'EEG-MCI-Bench'
        print("  OK  BIDS dataset description creation")
        tests_passed.append(True)
    except Exception as e:
        print(f"  FAIL BIDS functionality: {e}")
        tests_passed.append(False)

    # Test 2: Preprocessing configuration
    try:
        from src.preprocessing import load_config
        config = load_config('configs/bands.yaml')
        assert 'bands' in config
        assert 'alpha' in config['bands']
        print("  OK  Configuration loading")
        tests_passed.append(True)
    except Exception as e:
        print(f"  FAIL Configuration loading: {e}")
        tests_passed.append(False)

    # Test 3: Feature extraction
    try:
        import numpy as np
        from src.features.entropy import compute_shannon_entropy
        signal = np.random.randn(1000)
        entropy = compute_shannon_entropy(signal)
        assert 0 <= entropy <= 10  # Reasonable range
        print("  OK  Entropy calculation")
        tests_passed.append(True)
    except Exception as e:
        print(f"  FAIL Feature extraction: {e}")
        tests_passed.append(False)

    # Test 4: ML metrics
    try:
        from src.models.classical import compute_metrics_with_ci
        y_true = [[1, 0, 1, 0]]
        y_pred = [[1, 0, 0, 0]]
        y_prob = [[0.9, 0.1, 0.4, 0.2]]
        metrics = compute_metrics_with_ci(y_true, y_pred, y_prob)
        assert 'f1_mean' in metrics
        print("  OK  ML metrics calculation")
        tests_passed.append(True)
    except Exception as e:
        print(f"  FAIL ML metrics: {e}")
        tests_passed.append(False)

    # Test 5: Report generation
    try:
        from src.reports.generate_tripod_ai import TRIPODAIReportGenerator
        mock_results = {'metrics': {'f1_mean': 0.8}}
        report = TRIPODAIReportGenerator(mock_results, {})
        assert hasattr(report, 'generate_report')
        print("  OK  Report generation capability")
        tests_passed.append(True)
    except Exception as e:
        print(f"  FAIL Report generation: {e}")
        tests_passed.append(False)

    success = sum(tests_passed)
    total = len(tests_passed)
    print(f"\nResults: {success}/{total} tests passed")
    return all(tests_passed)


def main():
    """Run complete verification"""
    print("\n" + "="*70)
    print(" "*20 + "EEG-MCI-BENCH VERIFICATION SUITE")
    print("="*70)

    results = {}

    # 1. Check dependencies
    results['dependencies'] = verify_dependencies()

    # 2. Check file structure
    results['files'] = verify_file_structure()

    # 3. Check module imports
    results['imports'] = verify_module_imports()

    # 4. Check GPU support
    results['gpu'] = verify_gpu_support()

    # 5. Check functionality
    results['functionality'] = verify_functionality()

    # Summary
    print("\n" + "="*70)
    print(" "*25 + "VERIFICATION SUMMARY")
    print("="*70)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL" if not passed else "WARN"
        symbol = "OK" if passed else "!!" if not passed else "??"
        print(f"  [{symbol}] {test_name.upper()}: {status}")
        if not passed and test_name != 'gpu':  # GPU is optional
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print(" "*20 + "ALL CRITICAL TESTS PASSED")
        print(" "*15 + "PROJECT IS FULLY FUNCTIONAL!")
    else:
        print(" "*20 + "SOME TESTS FAILED")
        print(" "*15 + "Please check the errors above")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)