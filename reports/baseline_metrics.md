# Baseline Metrics — AD vs HC (LOSO, SVM)

- N samples: 4
- Bootstrap iterations: 2000

## Performance Metrics with 95% Confidence Intervals

- **F1 Score**: 0.000 (95% CI: [0.000, 0.000])
- **MCC**: -1.000 (95% CI: [-1.000, -1.000])
- **AUC**: 0.000 (95% CI: [0.000, 0.000])

## Confusion Matrices by Fold (0=HC,1=AD)
- Fold 1: TN=0, FP=1, FN=0, TP=0
- Fold 2: TN=0, FP=0, FN=1, TP=0
- Fold 3: TN=0, FP=1, FN=0, TP=0
- Fold 4: TN=0, FP=0, FN=1, TP=0

## Notes
- Confidence intervals computed using bootstrap resampling (2000 iterations)
- Each fold represents leave-one-subject-out cross-validation
- Poor performance due to minimal sample size (4 subjects)
