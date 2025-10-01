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
