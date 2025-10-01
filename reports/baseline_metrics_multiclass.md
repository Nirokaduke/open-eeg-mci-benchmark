# Multi-class Classification Report — AD/FTD/HC (LOSO, SVM)

## Overall Performance

- **Accuracy**: 0.000
- **Macro-F1**: 0.000 (95% CI: [0.000, 0.000])
- **Macro-AUC**: nan (95% CI: [0.000, 0.000])

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|}
| AD | 0.000 | 0.000 | 0.000 | 2 |
| FTD | 0.000 | 0.000 | 0.000 | 0 |
| HC | 0.000 | 0.000 | 0.000 | 2 |

## Confusion Matrix

```
       Predicted
       AD  FTD  HC
AD   [  0   0   2]
FTD  [  0   0   0]
HC   [  2   0   0]
```

## Class Distribution

- AD: 2 (50.0%)
- FTD: 0 (0.0%)
- HC: 2 (50.0%)

## Notes
- Class imbalance present (FTD has fewer samples)
- Confidence intervals computed using bootstrap (1000 iterations)
- LOSO-CV ensures subject-level independence
- Poor performance due to minimal sample size
