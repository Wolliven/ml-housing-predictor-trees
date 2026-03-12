# Baseline Analysis – Tree Models

## Overview

This analysis evaluates the baseline performance of tree-based regression models on the California Housing dataset.  
The goal is to understand how a single Decision Tree compares to a Random Forest ensemble when trained on the same feature set used in the previous linear model project.

Evaluation is performed using cross-validation predictions to estimate generalization performance.

---

# Metrics

## Decision Tree (depth=5)

R²: 0.4999  
MAE: 0.6039  
RMSE: 0.8160  

## Random Forest

R²: 0.6669  
MAE: 0.4788  
RMSE: 0.6660  

---

# Error Statistics

## Decision Tree

mean error: -0.0135  
standard deviation: 0.8159  
max error: 3.7215  
min error: -4.1709  

## Random Forest

mean error: 0.0306  
standard deviation: 0.6653  
max error: 2.9412  
min error: -3.7326  

---

# Worst Prediction Samples

## Decision Tree

The largest prediction errors for the Decision Tree occur in districts with unusual feature values, such as extremely large `AveRooms` or very small population counts.  

Example rows include districts with extremely high room averages (e.g. `AveRooms > 100`), which likely represent atypical census blocks or aggregated reporting artifacts.

These extreme values appear to cause unstable splits in the tree, producing large prediction errors.

## Random Forest

Random Forest also struggles with some of the same unusual districts, but the magnitude of the errors is smaller overall.

Because the final prediction is the average of many trees, extreme splits from individual trees tend to cancel out, producing more stable predictions.

---

# Observations

### Decision Tree Performance

The Decision Tree explains roughly **50% of the variance** in housing prices.  
While this indicates the model captures meaningful signal, it performs worse than the linear models from the previous project.

This suggests that a single decision tree is **highly sensitive to variance in the dataset**, particularly when extreme feature values are present.

The relatively large RMSE and error spread indicate that the tree may be overfitting specific patterns in the training folds.

---

### Random Forest Improvement

The Random Forest model significantly improves performance:

- R² increases from **~0.50 to ~0.67**
- RMSE drops from **0.816 → 0.666**
- MAE drops from **0.604 → 0.479**

This improvement is expected because Random Forest reduces model variance by averaging predictions from many decision trees trained on different bootstrap samples of the data.

The ensemble approach makes the model much more stable and less sensitive to individual noisy splits.

---

### Comparison with Linear Models

In the previous project, Linear Regression and Ridge Regression achieved approximately:

R² ≈ **0.57** and R² ≈ **0.61** with feature engineering.

The Random Forest baseline now reaches:

R² ≈ **0.67**

This indicates that tree-based ensemble models capture **non-linear relationships and feature interactions** that linear models cannot represent.

---

### Error Distribution

Both models show mean prediction errors close to zero, suggesting there is **no strong overall bias toward overprediction or underprediction**.

However, large individual errors still appear for districts with unusual feature values, especially where variables such as `AveRooms` or `AveOccup` take extreme values.

These outliers likely represent irregular census tracts and may require additional analysis or feature engineering.

---

# Key Takeaways

1. A single Decision Tree provides moderate predictive performance but suffers from high variance and unstable splits.
2. Random Forest substantially improves performance by averaging many trees, reducing variance and producing more reliable predictions.
3. Tree-based models outperform the previous linear baseline, indicating that the dataset contains important **non-linear relationships**.
4. Extreme feature values appear frequently among the worst predictions and may require further investigation.

---

# Next Steps

Further analysis could explore:

- Decision tree depth experiments to study overfitting behavior
- Feature importance analysis from the Random Forest model
- Feature engineering based on household ratios (e.g., rooms per household)
- Residual analysis to better understand large prediction errors

These experiments should help clarify which variables contribute most to prediction accuracy and how model complexity affects generalization.