# Tree Models Analysis

## 1. Overview
This document analyzes the behavior of tree-based regression models on the California Housing dataset.

The goal of this analysis is to understand how Decision Trees and Random Forest models behave, how model complexity affects performance, and how these models compare with the linear models implemented in the previous project.

Experiments in this report focus on:

- Understanding the behavior of decision trees
- Studying the effect of tree depth on model performance
- Visualizing tree structures
- Comparing single trees with Random Forest ensembles
- Identifying important features driving model predictions

All evaluations are performed using cross-validation in order to estimate generalization performance.

---

## 2. Baseline Results
Before conducting further experiments, baseline performance was measured using a Decision Tree with limited depth and a Random Forest model trained on the same feature set used in the previous linear model project.

### 2.1 Decision Tree (depth = 5)

R²: 0.4999  
MAE: 0.6039  
RMSE: 0.8160  

### 2.2 Random Forest

R²: 0.6669  
MAE: 0.4788  
RMSE: 0.6660  

### 2.3 Initial Observations

The Decision Tree achieves moderate predictive performance but remains less accurate than the Random Forest model.

Random Forest significantly improves performance by combining predictions from many trees, reducing variance and producing more stable results.

Compared with the previous linear models (R² ≈ 0.57–0.61), the Random Forest model achieves stronger performance, suggesting that tree-based methods capture non-linear relationships and feature interactions that linear models cannot represent.

Further experiments will analyze how tree complexity affects model behavior and investigate the internal structure of the learned trees.

---

## 3. Decision Tree Experiments

### 3.1 Decision Tree Depth vs Performance

To study how model complexity affects performance, the maximum depth of the Decision Tree was varied from 1 to 30.  
Each configuration was evaluated using 5-fold cross-validation, recording the mean R² score and the standard deviation across folds.

![Decision Tree performance vs depth](tree_depth_experiment.png)

#### Observations

The results show a clear relationship between tree depth and model performance.

For very shallow trees (depth 1–3), performance is relatively low. This indicates **underfitting**, as the model is too simple to capture the relationships in the dataset.

As the tree depth increases, the mean R² score improves significantly, reaching its highest values around **depth 6–8**. In this range, the model has enough flexibility to capture useful patterns in the data without excessively fitting noise.

Beyond this point, performance gradually declines as depth increases further. Deeper trees introduce many additional splits, which can cause the model to learn highly specific patterns from the training folds that do not generalize well to unseen data. This behavior is consistent with **overfitting**.

The standard deviation across cross-validation folds also remains relatively stable but tends to increase slightly for deeper trees, suggesting greater sensitivity to the specific training subset used in each fold.

#### Interpretation

This experiment illustrates the classical **bias–variance tradeoff**:

- **Low depth** → high bias, underfitting  
- **Moderate depth** → best balance between bias and variance  
- **High depth** → increased variance and overfitting  

The optimal region for this dataset appears to be around **depth 6–8**, which aligns with the earlier baseline experiment where a depth-5 tree produced reasonable but not optimal results.

These findings confirm that controlling tree complexity is critical when using single decision trees, as unrestricted trees can easily grow large and memorize training data rather than learning generalizable patterns.

### 3.2 Tree Visualization
(tree diagram)

## 4. Random Forest Analysis

### 4.1 Feature Importance
### 4.2 Comparison with Decision Trees

## 5. Discussion

## 6. Key Takeaways

## 7. Next Steps