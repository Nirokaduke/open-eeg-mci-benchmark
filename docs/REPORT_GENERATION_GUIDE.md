# Report Generation Guide for EEG-MCI-Bench

## Overview

This guide describes the three comprehensive report generation modules designed to produce publication-ready documentation following international reporting standards.

## Implementation Summary

### Module Statistics
- **Total Lines of Code**: 2,222
- **Three Complete Modules**:
  1. `generate_tripod_ai.py` (651 lines) - ML model transparency
  2. `generate_stard.py` (896 lines) - Diagnostic accuracy
  3. `evidence_map.py` (675 lines) - Evidence synthesis

All modules have been validated and compile successfully.

---

## Quick Start

### 1. Basic Usage

Generate all three report types:

```bash
# TRIPOD+AI Report
python src/reports/generate_tripod_ai.py \
    --results-dir data/derivatives/results \
    --config configs/baseline_config.json \
    --output-dir reports

# STARD Report
python src/reports/generate_stard.py \
    --results-dir data/derivatives/results \
    --config configs/baseline_config.json \
    --output-dir reports

# Evidence Synthesis
python src/reports/evidence_map.py \
    --results-dir data/derivatives/results \
    --output-dir reports \
    --literature-data data/literature/analyzed_resources.tab
```

### 2. Expected Outputs

**TRIPOD+AI Report:**
- ✅ Complete markdown report with 26 checklist items
- ✅ ROC curves comparing all models
- ✅ Confusion matrices
- ✅ Feature importance plots
- ✅ Fairness analysis visualizations

**STARD Report:**
- ✅ Complete markdown report with 29 checklist items
- ✅ CONSORT-style participant flow diagram
- ✅ Diagnostic accuracy tables (sensitivity, specificity, PPV, NPV)
- ✅ ROC curves with confidence bands
- ✅ Confusion matrices with percentages
- ✅ Cross-tabulation (2×2 tables)

**Evidence Synthesis:**
- ✅ Model comparison charts (F1, AUC, MCC, Balanced Accuracy)
- ✅ Forest plots with 95% confidence intervals
- ✅ Aggregated feature importance across models
- ✅ Performance heatmaps
- ✅ Interactive HTML dashboard
- ✅ Summary tables (CSV + PNG)
- ✅ Literature comparison (optional)

---

## Module Details

### 1. TRIPOD+AI Report Generator

**Purpose**: Transparent reporting of machine learning prediction models in healthcare

**Compliance**: TRIPOD+AI Statement (Collins et al., 2024, BMJ)

**Key Sections Generated**:
1. Title and Abstract (Items 1-2)
2. Introduction - Background and objectives (Item 3)
3. Methods:
   - Source of data (Items 4a-4b)
   - Participants (Items 5a-5c)
   - Outcome definition (Items 6a-6b)
   - Predictors/features (Items 7a-7b)
   - Sample size (Item 8)
   - Missing data (Item 9)
   - Statistical analysis (Items 10a-10d)
   - AI-specific methods (Items 10e-10h)
4. Results:
   - Participants flow (Item 11)
   - Model development (Items 12-14)
   - Model performance with CI (Items 15-16)
5. Discussion (Items 17-19)
6. Other information (Items 20-22)
7. Complete checklist table

**Visualizations**:
- ROC curves with AUC values
- Confusion matrices
- Feature importance (top 20)
- Fairness analysis by demographic subgroups

**Clinical Focus**: Emphasizes model transparency, generalizability, and clinical applicability

---

### 2. STARD Report Generator

**Purpose**: Diagnostic accuracy study reporting for EEG-based MCI classification

**Compliance**: STARD 2015 Guidelines (Bossuyt et al., 2015, BMJ)

**Key Sections Generated**:
1. Title and Abstract (Items 1-2)
2. Introduction - Study rationale (Item 3)
3. Methods:
   - Study design (Item 4)
   - Participant recruitment (Item 5)
   - Eligibility criteria (Item 6)
   - Index test description (Items 7-10)
   - Reference standard (Items 11-13)
   - Sample size calculation (Item 14)
   - Missing data handling (Item 15)
   - Statistical methods (Items 16-18)
4. Results:
   - Participant flow diagram (Item 19)
   - Baseline characteristics (Item 20)
   - Diagnostic accuracy estimates (Items 21-23)
   - Timing and adverse events (Items 24-25)
5. Discussion - Clinical applicability (Item 26)
6. Complete STARD checklist

**Visualizations**:
- CONSORT-style participant flow diagram
- ROC curves labeled with sensitivity/specificity axes
- Confusion matrices with counts and percentages
- Diagnostic accuracy tables with likelihood ratios

**Clinical Focus**: Emphasizes diagnostic test performance, patient flow, and clinical implementation

---

### 3. Evidence Map Generator

**Purpose**: Visual synthesis and meta-analysis of results across models and literature

**Key Visualizations**:

1. **Model Comparison Charts** (4 panels):
   - F1 Score comparison
   - AUC comparison
   - MCC comparison
   - Balanced Accuracy comparison
   - Best model highlighted in green
   - Error bars showing 95% CI

2. **Forest Plots** (4 metrics):
   - F1 Score
   - AUC
   - Sensitivity
   - Specificity
   - Horizontal CI bars
   - Reference line at mean value

3. **Feature Importance Synthesis**:
   - Aggregates importance across all models
   - Mean ± SD for each feature
   - Top 25 features displayed
   - Horizontal bar chart

4. **Performance Heatmap**:
   - All metrics × all models
   - Color-coded (red-yellow-green)
   - Annotated with percentage values

5. **Interactive Dashboard** (HTML):
   - 4-panel Plotly dashboard
   - F1 and AUC bar charts with error bars
   - Sensitivity vs. Specificity scatter plot
   - Radar chart for best model
   - Interactive hover tooltips

6. **Summary Table**:
   - Ranked by F1 score
   - All metrics with confidence intervals
   - CSV format for further analysis
   - Styled PNG for publication

7. **Literature Comparison** (optional):
   - Current study vs. published literature
   - Sample size vs. accuracy scatter
   - Interactive HTML version
   - Context for study findings

**Output Formats**:
- Static PNG images (publication-ready, 300 DPI)
- Interactive HTML (Plotly with CDN)
- CSV tables (for meta-analysis)

---

## Input Requirements

### 1. Model Results Format

Each model should have a JSON file: `{model_name}_results.json`

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
    "sensitivity_ci_lower": 0.820,
    "sensitivity_ci_upper": 0.910,
    "specificity": 0.833,
    "specificity_ci_lower": 0.780,
    "specificity_ci_upper": 0.885,
    "accuracy": 0.850,
    "balanced_accuracy": 0.850,
    "ppv": 0.833,
    "npv": 0.867
  },
  "confusion_matrix": [[25, 5], [4, 26]],
  "fpr": [0.0, 0.05, 0.1, 0.15, ..., 1.0],
  "tpr": [0.0, 0.35, 0.55, 0.72, ..., 1.0],
  "feature_importance": {
    "alpha_power_frontal": 0.152,
    "theta_power_temporal": 0.118,
    "complexity_entropy": 0.105,
    "connectivity_plv": 0.098,
    "delta_power_parietal": 0.087
  }
}
```

### 2. Configuration File

JSON or YAML file with study metadata:

```json
{
  "study_period": "2020-2024",
  "location": "Multi-center (3 sites)",
  "eeg_system": "BrainVision actiCHamp Plus 32-channel",
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
  "models": ["SVM", "Random Forest", "XGBoost", "LightGBM"],
  "hyperparam_method": "Grid search with 5-fold CV",
  "imbalance_method": "SMOTE",
  "ci_method": "Bootstrap (1000 iterations)",
  "python_version": "3.11",
  "sklearn_version": "1.3",
  "mne_version": "1.5",
  "random_seed": 42,
  "code_repository": "https://github.com/yourname/eeg-mci-bench",
  "ethics_approval": "IRB Protocol #2023-001",
  "registration": "ClinicalTrials.gov NCT12345678"
}
```

### 3. Data Summary (Optional)

`data_summary.json`:

```json
{
  "screened": 150,
  "eligible": 120,
  "enrolled": 100,
  "completed_index": 95,
  "completed_reference": 95,
  "after_qc": 88,
  "final_n": 88,
  "mci_n": 44,
  "hc_n": 44,
  "excluded_screening": 30,
  "excluded_qc": 7,
  "mci_age_mean": 72.5,
  "mci_age_sd": 6.2,
  "hc_age_mean": 68.3,
  "hc_age_sd": 5.8,
  "mci_female_n": 22,
  "mci_female_pct": 50,
  "hc_female_n": 23,
  "hc_female_pct": 52.3,
  "age_pvalue": "0.02",
  "sex_pvalue": "0.85"
}
```

---

## Workflow Integration

### Recommended Pipeline

```bash
# Step 1: Train models and save results
python code/04_train_models.py --config configs/baseline_config.json

# Step 2: Generate all three reports
python src/reports/generate_tripod_ai.py \
    --results-dir data/derivatives/results \
    --config configs/baseline_config.json \
    --output-dir reports

python src/reports/generate_stard.py \
    --results-dir data/derivatives/results \
    --config configs/baseline_config.json \
    --output-dir reports

python src/reports/evidence_map.py \
    --results-dir data/derivatives/results \
    --output-dir reports \
    --literature-data data/literature/analyzed_resources.tab

# Step 3: Review outputs
ls -lh reports/
```

### Automation with Make

Add to `Makefile`:

```makefile
.PHONY: reports
reports:
	python src/reports/generate_tripod_ai.py \
	    --results-dir data/derivatives/results \
	    --config configs/baseline_config.json \
	    --output-dir reports
	python src/reports/generate_stard.py \
	    --results-dir data/derivatives/results \
	    --config configs/baseline_config.json \
	    --output-dir reports
	python src/reports/evidence_map.py \
	    --results-dir data/derivatives/results \
	    --output-dir reports
	@echo "All reports generated in reports/"
```

---

## Customization

### Adding New Metrics

Edit the metric lists in each module:

**TRIPOD+AI** (`generate_tripod_ai.py`):
```python
metrics = result.get('test_metrics', {})
# Add your custom metric
custom_metric = metrics.get('custom_metric', 0)
```

**STARD** (`generate_stard.py`):
```python
# Add to diagnostic accuracy table
table += f"| {model_name} | ... | {custom_metric:.2f} |\n"
```

**Evidence Map** (`evidence_map.py`):
```python
# Add to metrics list in performance heatmap
metric_cols = ['f1', 'auc', 'mcc', 'custom_metric']
```

### Changing Visualizations

**Matplotlib style**:
```python
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("Set2")
```

**Figure size**:
```python
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust width, height
```

**Color scheme**:
```python
colors = ['steelblue', 'coral', 'forestgreen', 'orchid']
```

---

## Troubleshooting

### Common Issues

**1. Missing confidence intervals**
```
Problem: CI columns not in results
Solution: Compute CIs during model evaluation or use bootstrap estimates
```

**2. Literature data not loading**
```
Problem: Column names not recognized
Solution: Check column naming in literature file. The smart picker handles variations but may need manual specification
```

**3. Font size too small**
```
Problem: Text unreadable in figures
Solution: Adjust sns.set_context("paper", font_scale=1.5)
```

**4. Memory issues with large datasets**
```
Problem: OOM when generating figures
Solution: Reduce DPI (dpi=150 instead of 300) or process models sequentially
```

---

## Best Practices

### For Publication

1. **Run all three generators** - Complementary perspectives required for comprehensive reporting

2. **Version control** - Tag report versions with git:
   ```bash
   git add reports/
   git commit -m "Add TRIPOD+AI, STARD, and Evidence Map reports v1.0"
   git tag -a reports-v1.0 -m "Initial publication reports"
   ```

3. **Archive configuration** - Save exact config used:
   ```bash
   cp configs/baseline_config.json reports/config_used.json
   ```

4. **Check completeness** - Review checklists in TRIPOD+AI and STARD reports

5. **Peer review** - Have coauthors review reports before submission

### For Reproducibility

1. **Document software versions** - Include in config file

2. **Share code and configs** - Make repository public or share via OSF

3. **Provide example data** - Include de-identified subset for testing

4. **Document deviations** - Note any deviations from standard protocols

---

## Citation

When using these report generators in publications, cite:

**TRIPOD+AI:**
> Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378

**STARD:**
> Bossuyt PM, Reitsma JB, Bruns DE, et al. STARD 2015: an updated list of essential items for reporting diagnostic accuracy studies. BMJ 2015;351:h5527. doi:10.1136/bmj.h5527

**Software:**
> [Your Citation]. EEG-MCI-Bench Report Generation Modules. Available at: https://github.com/yourname/eeg-mci-bench

---

## Support

- **Documentation**: See `src/reports/README.md`
- **Examples**: Check `reports/examples/` for sample outputs
- **Issues**: Open GitHub issue with:
  - Error message
  - Configuration file
  - Minimal reproducible example

---

**Last Updated**: 2025-10-01
**Version**: 1.0.0
**Maintainer**: EEG-MCI-Bench Development Team
