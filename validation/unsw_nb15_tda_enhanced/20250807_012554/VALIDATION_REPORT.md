# Optimized UNSW-NB15 TDA Validation Results

**Timestamp:** 20250807_013125
**Target:** 0.75 F1-Score
**Baseline:** 0.567 F1-Score

## Results

**Best Model:** Random Forest
**Best F1-Score:** 0.902
**vs Target:** +0.152
**vs Baseline:** +0.335 (+59.1%)

## Model Performance

### Logistic Regression
- **F1-Score**: 0.844
- **Accuracy**: 0.805
- **Precision**: 0.905
- **Recall**: 0.791
- **ROC-AUC**: 0.884
- **Cross-Validation**: 0.838 ± 0.004
- **Training Time**: 0.0s
- **Confusion Matrix**: TP=2372, FP=249, FN=628, TN=1251

### Random Forest
- **F1-Score**: 0.902
- **Accuracy**: 0.873
- **Precision**: 0.931
- **Recall**: 0.874
- **ROC-AUC**: 0.959
- **Cross-Validation**: 0.903 ± 0.005
- **Training Time**: 0.2s
- **Confusion Matrix**: TP=2623, FP=194, FN=377, TN=1306


## Files Generated
- Confusion Matrix: `plots/confusion_matrix_20250807_013125.png`
- Model Comparison: `plots/model_comparison_20250807_013125.png`  
- Results Data: `data/results_20250807_013125.json`
