# ML Housing Price Predictor – Tree Models

A structured machine learning project focused on learning and evaluating tree-based regression models through a disciplined, reproducible training workflow.

This project builds upon previous work with linear and regularized regression models and introduces a new family of machine learning models: **decision trees and ensemble methods**. The goal is to understand how tree models capture non-linear relationships, how they behave under different levels of complexity, and how they compare to linear approaches on the same dataset.

The emphasis of this project is **model intuition, evaluation rigor, and controlled experimentation** rather than feature expansion or system complexity.

---

## Project Objectives

This project aims to implement a production-style machine learning workflow that:

1. Loads structured housing data
2. Trains and evaluates tree-based regression models
3. Compares a single decision tree with a random forest
4. Studies model complexity and overfitting behavior
5. Selects the best-performing model programmatically
6. Saves a fully self-contained model artifact
7. Performs inference with strict feature validation

The project prioritizes **interpretability, experimentation discipline, and deeper understanding of non-linear models**.

---

## Core Learning Goals

This project is designed to strengthen understanding of several important machine learning concepts:

### Non-Linear Modeling

Unlike linear models, which represent relationships as weighted sums of features, tree models learn **rule-based decision boundaries** by recursively splitting the feature space. This allows them to capture more complex patterns in the data.

### Decision Tree Splitting Logic

Decision trees learn thresholds such as:

- `MedInc < threshold`
- `Latitude > threshold`

These splits partition the data into regions with different predicted values.

### Overfitting and Model Complexity

Decision trees provide a clear environment for observing bias–variance tradeoffs:

- Shallow trees tend to **underfit**
- Deep trees tend to **overfit**

Hyperparameters such as `max_depth`, `min_samples_split`, and `min_samples_leaf` control this behavior.

### Ensemble Learning

Random forests improve stability and performance by combining **many independent decision trees** trained on different samples of the data. This demonstrates a core machine learning idea:

> Multiple weak, unstable models can combine to form a stronger and more robust predictor.

### Feature Importance

Tree-based models provide built-in feature importance estimates, allowing inspection of which variables contribute most to predictions.

---

## Candidate Models

This project evaluates the following models:

- `DecisionTreeRegressor`
- `RandomForestRegressor`

Both models are implemented using **scikit-learn** and evaluated using consistent cross-validation procedures.

---

## Training Workflow

Model training follows a structured workflow:

1. Load dataset
2. Separate features and target variable
3. Build candidate models
4. Evaluate models using cross-validation
5. Compare mean and variance of performance metrics
6. Select the best-performing model
7. Train the final model on the full dataset
8. Save a structured model artifact containing:
   - trained model
   - feature schema
   - metadata
   - training configuration

This workflow emphasizes **reproducibility and consistent evaluation**.

---

## Analysis and Experiments

The project includes dedicated analysis scripts that explore:

- Baseline decision tree performance
- Effects of tree depth on overfitting
- Hyperparameter experimentation
- Feature importance inspection
- Performance comparison with previous linear and Ridge models

Experiment outputs and notes are stored in the `reports/` directory.

---

## Project Structure

```

ml-housing-predictor-trees/
├── data/
├── input/
├── reports/
├── scripts/
├── train.py
├── predict.py
├── ml_engine.py
├── requirements.txt
├── README.md
└── .gitignore

```

---

## Scope

This project intentionally focuses on **classical machine learning using tree-based models**.

Out of scope:

- Deep learning
- Neural networks
- API deployment
- Distributed training
- Advanced boosting libraries such as XGBoost or LightGBM

These topics may be explored in later projects.

---

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Philosophy

This project prioritizes:

* Understanding over speed
* Controlled experimentation over blind hyperparameter tuning
* Reproducibility over ad-hoc experimentation
* Engineering discipline over notebook-style workflows
