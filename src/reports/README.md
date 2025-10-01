# EEG-MCI-Bench Report Generation Modules

This directory contains three comprehensive report generation tools that produce publication-ready documentation following international guidelines.

## Modules

### 1. `generate_tripod_ai.py` - TRIPOD+AI Compliant Reports

Generates transparent reporting following TRIPOD+AI (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis + Artificial Intelligence) guidelines.

**Usage:**
```bash
python src/reports/generate_tripod_ai.py \
    --results-dir data/derivatives/results \
    --config configs/study_config.json \
    --output-dir reports \
    --title "EEG-Based MCI Classification"
```

**Outputs:**
- `TRIPOD_AI_report_YYYYMMDD_HHMMSS.md` - Complete markdown report
- `roc_curves.png` - ROC curves for all models
- `confusion_matrices.png` - Confusion matrices
- `feature_importance.png` - Top discriminative features

**Key Features:**
- Complete TRIPOD+AI checklist (26 items)
- Performance metrics with 95% confidence intervals
- Feature importance analysis
- Fairness and bias assessment
- Publication-ready figures

---

### 2. `generate_stard.py` - STARD 2015 Compliant Reports

Generates diagnostic accuracy study reports following STARD 2015 (Standards for Reporting of Diagnostic Accuracy Studies) guidelines.

**Usage:**
```bash
python src/reports/generate_stard.py \
    --results-dir data/derivatives/results \
    --config configs/study_config.json \
    --output-dir reports \
    --title "EEG-Based MCI Diagnostic Accuracy Study"
```

**Outputs:**
- `STARD_report_YYYYMMDD_HHMMSS.md` - Complete markdown report
- `stard_flow_diagram.png` - Participant flow diagram
- `stard_roc_curves.png` - Diagnostic performance ROC curves
- `stard_confusion_matrices.png` - Confusion matrices with percentages

**Key Features:**
- Complete STARD 2015 checklist (29 items)
- Participant flow diagram
- Sensitivity, specificity, PPV, NPV with 95% CI
- Likelihood ratios
- Cross-tabulation (2×2 tables)
- Clinical applicability assessment

---

### 3. `evidence_map.py` - Evidence Synthesis & Meta-Analysis

Creates comprehensive visualizations synthesizing results across multiple models and comparing with literature.

**Usage:**
```bash
python src/reports/evidence_map.py \
    --results-dir data/derivatives/results \
    --output-dir reports \
    --literature-data data/literature/analyzed_resources.tab
```

**Outputs:**
- `model_comparison.png` - Bar charts comparing all metrics
- `forest_plots.png` - Forest plots with 95% CI
- `feature_importance_synthesis.png` - Aggregated feature importance
- `performance_heatmap.png` - Heatmap of all metrics
- `evidence_dashboard.html` - Interactive Plotly dashboard
- `performance_summary.csv` - Tabular summary data
- `summary_table.png` - Formatted publication table
- `literature_comparison.png` - Current study vs. literature (if provided)
- `evidence_map.html` - Interactive literature comparison (if provided)

**Key Features:**
- Model comparison across all metrics
- Forest plots for meta-analysis visualization
- Feature importance aggregation across models
- Interactive dashboards (Plotly)
- Literature comparison (optional)
- Publication-ready figures

---

## Configuration File Format

All report generators require a configuration file (JSON or YAML) with study metadata:

```json
{
  "study_period": "2020-2024",
  "location": "Multi-center",
  "eeg_system": "BrainVision actiCHamp Plus",
  "sampling_rate": 1000,
  "n_channels": 32,
  "reference": "Average reference",
  "recording_duration": 5,
  "bandpass_filter": "0.5-45",
  "line_noise": 50,
  "reref_method": "Average reference",
  "artifact_method": "Automated threshold + ICA",
  "ica_method": "Extended Infomax",
  "epoch_length": 2,
  "epoch_overlap": 50,
  "min_epochs": 30,
  "models": ["SVM", "Random Forest", "XGBoost"],
  "python_version": "3.11",
  "random_seed": 42,
  "code_repository": "https://github.com/yourname/eeg-mci-bench"
}
```

## Input Data Format

### Model Results

Results should be in JSON format with the following structure:

```json
{
  "test_metrics": {
    "f1": 0.850,
    "f1_ci_lower": 0.820,
    "f1_ci_upper": 0.880,
    "auc": 0.920,
    "auc_ci_lower": 0.895,
    "auc_ci_upper": 0.945,
    "mcc": 0.698,
    "sensitivity": 0.867,
    "specificity": 0.833,
    "accuracy": 0.850,
    "balanced_accuracy": 0.850,
    "ppv": 0.833,
    "npv": 0.867
  },
  "confusion_matrix": [[25, 5], [4, 26]],
  "fpr": [0, 0.1, 0.2, ..., 1.0],
  "tpr": [0, 0.3, 0.5, ..., 1.0],
  "feature_importance": {
    "alpha_power_frontal": 0.15,
    "theta_power_temporal": 0.12,
    ...
  }
}
```

Save results as: `{results_dir}/{model_name}_results.json`

### Data Summary (Optional)

Create `{results_dir}/data_summary.json`:

```json
{
  "screened": 120,
  "enrolled": 100,
  "final_n": 88,
  "mci_n": 44,
  "hc_n": 44,
  "mci_age_mean": 72.5,
  "mci_age_sd": 6.2,
  "hc_age_mean": 68.3,
  "hc_age_sd": 5.8,
  "mci_female_pct": 50,
  "hc_female_pct": 52
}
```

### Literature Data (Optional)

Tab-separated or comma-separated file with columns like:
- `accuracy` or `acc` or `performance`
- `n` or `n_subjects` or `sample_size`
- Additional metadata columns

The evidence map generator uses smart column detection to handle various naming conventions.

---

## Best Practices

1. **Run All Three Generators**: Each provides complementary perspectives
   - TRIPOD+AI: ML model development transparency
   - STARD: Diagnostic accuracy assessment
   - Evidence Map: Visual synthesis and comparison

2. **Complete Metadata**: Provide comprehensive configuration to maximize report completeness

3. **Multiple Models**: Test 3-5 different algorithms for robust comparison

4. **Literature Context**: Include literature data for evidence_map.py to contextualize findings

5. **Review Before Submission**: All reports include checklists - verify completeness

---

## Dependencies

Required Python packages:
```
numpy>=1.21
pandas>=1.3
matplotlib>=3.5
seaborn>=0.11
scipy>=1.7
plotly>=5.0
scikit-learn>=1.0
pyyaml>=6.0
```

Install via:
```bash
pip install -r requirements.txt
```

---

## Citation

If you use these report generators, please cite:

**TRIPOD+AI:**
> Collins GS, et al. (2024). TRIPOD+AI Statement. BMJ 385:e078378.

**STARD:**
> Bossuyt PM, et al. (2015). STARD 2015: An Updated List of Essential Items for Reporting Diagnostic Accuracy Studies. BMJ 351:h5527.

---

## Support

For issues or questions:
1. Check configuration file format
2. Verify input data structure matches expected format
3. Review example outputs in `reports/examples/`
4. Open an issue on GitHub with error messages and minimal reproducible example

---

Last updated: 2025-10-01
