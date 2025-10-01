# EEG-MCI-Bench

A reproducible benchmark and research toolkit for MCI detection from EEG/ERP.

## Goals
- Standardize datasets in **BIDS-EEG** format
- Provide feature pipelines (spectrum, complexity, connectivity, ERP)
- Enforce **subject-level** validation (LOSO) and optional external test set
- Report **F1**, **MCC**, **AUC** with 95% CI and reproducible configs
- Generate transparent reports aligned with **TRIPOD+AI** and **STARD**

## Quick start
1. Create and activate a Python 3.11+ virtual environment.
2. `pip install -r requirements.txt`
3. Organize raw data under `data/bids_raw/` (BIDS layout).
4. Run BIDS validation report:
   ```bash
   python code/convert_to_bids.py --validate
   ```
5. Preprocess & extract features:
   ```bash
   python code/preprocessing.py --config configs/bands.yaml
   python code/features/spectrum.py --config configs/bands.yaml
   python code/features/entropy.py --config configs/bands.yaml
   python code/features/connectivity.py --config configs/connectivity.yaml
   python code/features/erp.py --config configs/erp.yaml
   ```
6. Baselines (subject-level LOSO):
   ```bash
   python code/models/classical.py --cv loso --metrics f1 mcc auc
   ```
7. Deep learning (optional):
   ```bash
   python code/models/deep.py --cv loso --epochs 40
   ```
8. Generate reports:
   ```bash
   python code/reports/generate_tripod_ai.py
   python code/reports/generate_stard.py
   ```

See `CLAUDE.md` and `.claude/commands/` for Claude Code workflows/prompts.

## Quick Start (ds004504)

This benchmark includes support for the **ds004504** dataset (88 subjects with resting-state EEG):

```bash
# 1. Validate BIDS structure
python code/scripts/preview_labels.py --participants data/bids_raw/ds004504/participants.tsv

# 2. Preprocess subjects (example: first 5)
python code/preprocessing.py --config configs/bands.yaml --max-subjects 5

# 3. Extract features
python code/features/spectrum.py --config configs/bands.yaml
python code/features/connectivity.py --config configs/connectivity.yaml

# 4. Run baseline model with confidence intervals
python code/models/classical.py --participants data/bids_raw/ds004504/participants.tsv \
  --features data/derivatives/features.parquet --bootstrap 2000

# 5. Run multi-class classification (AD/FTD/HC)
python code/models/classical_multiclass.py --model svm \
  --output reports/baseline_metrics_multiclass.md
```

### Dataset Information
- **Subjects**: 88 (AD, FTD, HC groups)
- **Paradigm**: Eyes-closed resting-state
- **Channels**: 19 (10-20 system)
- **Sampling Rate**: 256 Hz
- **Label Mapping**: A→AD, F→FTD, C→HC

## Evidence Map

We provide an interactive visualization of 278 MCI-EEG studies from literature:

📊 **[View Evidence Map](reports/evidence_map.html)**

The evidence map includes:
- Sample sizes and accuracy metrics from published studies
- Interactive scatter plot visualization
- Literature references for reproducibility

Generate or update the evidence map:
```bash
python code/reports/evidence_map.py \
  --inputs data/literature/analyzed_resources_1_139.tab \
           data/literature/analyzed_resources_140_277.tab \
  --out reports/evidence_map.html
```

## Reports & Documentation

All analysis reports follow standardized formats:

- **[Acceptance Summary](reports/ACCEPTANCE_SUMMARY.md)** - Complete validation report (88% passed)
- **[TRIPOD+AI Report](reports/tripod_ai_report.md)** - ML model reporting standards
- **[STARD 2015 Report](reports/stard_2015_report.md)** - Diagnostic accuracy standards
- **[Baseline Metrics](reports/baseline_metrics.md)** - Model performance with 95% CI
- **[Multi-class Results](reports/baseline_metrics_multiclass.md)** - Three-class classification

## Project Structure

```
open-eeg-mci-benchmark/
├── code/               # Source code
│   ├── features/      # Feature extraction modules
│   ├── models/        # ML models (classical & deep)
│   ├── reports/       # Report generators
│   └── utils/         # Utilities
├── configs/           # Configuration files
├── data/
│   ├── bids_raw/     # Raw BIDS data
│   ├── derivatives/  # Processed features
│   └── literature/   # Literature .tab files
├── reports/          # Generated reports
└── requirements.txt  # Python dependencies
```

## Citation

If you use this benchmark, please cite:

```bibtex
@software{eeg_mci_benchmark_2025,
  title = {EEG-MCI-Benchmark: A BIDS-compliant Pipeline for MCI Detection},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/thc1006/open-eeg-mci-benchmark}
}
```

## License

CC0 1.0 Universal (CC0 1.0) - Public Domain Dedication
See [LICENSE](LICENSE) for details.
