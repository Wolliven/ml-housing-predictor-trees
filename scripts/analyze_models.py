"""
This script contains functions to analyze the performance of the decision tree and random forest models, including error analysis, feature importance, and visualizations.
The functions can be called to generate reports and visualizations that help understand the strengths and weaknesses of each model, as well as the importance of different features in the dataset.
The analysis includes:
- Baseline model evaluation using cross-validation predictions
- Error analysis to identify patterns in the prediction errors
- Experimenting with different tree depths to find the optimal depth for the decision tree model
- Visualizing the structure of a decision tree
- Analyzing feature importance in the random forest model
To run the analysis, simply execute this script. The generated reports and visualizations will be saved in the "reports" directory.
"""

from numpy import sqrt
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from ml_engine import load_dataset, build_models, add_features
from sklearn.tree import plot_tree

#Baseline model evaluation using cross-validation predictions
X, y = load_dataset("data/california_housing.csv")
model_tree, model_forest = build_models().values()

def analyze_models(X : pd.DataFrame, y : pd.Series, model_tree, model_forest) -> None:
    pred_forest = cross_val_predict(model_forest, X, y, cv=5)
    pred_tree = cross_val_predict(model_tree, X, y, cv=5)

    r2_tree = r2_score(y, pred_tree)
    r2_forest = r2_score(y, pred_forest)
    rmse_tree = sqrt(mean_squared_error(y, pred_tree))
    rmse_forest = sqrt(mean_squared_error(y, pred_forest))
    mae_tree = mean_absolute_error(y, pred_tree)
    mae_forest = mean_absolute_error(y, pred_forest)

    errors_tree = pred_tree - y
    errors_forest = pred_forest - y
    abs_errors_tree = abs(errors_tree)
    abs_errors_forest = abs(errors_forest)

    worst_tree_idx = abs_errors_tree.sort_values(ascending=False).head(10).index
    worst_forest_idx = abs_errors_forest.sort_values(ascending=False).head(10).index

    print(f"Decision Tree - R²: {r2_tree:.4f}, RMSE: {rmse_tree:.4f}, MAE: {mae_tree:.4f}")
    print(f"Random Forest - R²: {r2_forest:.4f}, RMSE: {rmse_forest:.4f}, MAE: {mae_forest:.4f}")

    print("\nDecision Tree error stats:")
    print("mean:", errors_tree.mean())
    print("std:", errors_tree.std())
    print("min:", errors_tree.min())
    print("max:", errors_tree.max())

    print("\nRandom Forest error stats:")
    print("mean:", errors_forest.mean())
    print("std:", errors_forest.std())
    print("min:", errors_forest.min())
    print("max:", errors_forest.max())

    print("\nWorst Decision Tree predictions:")
    print(X.loc[worst_tree_idx])
    print(y.loc[worst_tree_idx])

    print("\nWorst Random Forest predictions:")
    print(X.loc[worst_forest_idx])
    print(y.loc[worst_forest_idx])

#Baseline model evaluation
#analyze_models(X, y, model_tree, model_forest)

#Features
#X = add_features(X)

def analyze_tree(X : pd.DataFrame, y : pd.Series) -> None:
    mean_scores = []
    std_scores = []
    depths = list(range(1, 31))
    for depth in depths:
        tree = build_models(tree_depth=depth).get("decision_tree")
        scores = cross_val_score(tree, X, y, cv=5, scoring="r2")
        mean_scores.append(scores.mean())
        std_scores.append(scores.std())
    plt.figure(figsize=(8,5))

    plt.plot(depths, mean_scores, label="Mean R²")
    plt.plot(depths, std_scores, label="Std deviation")

    plt.xlabel("Tree depth")
    plt.ylabel("Score")
    plt.title("Decision Tree performance vs depth")

    plt.legend()
    plt.savefig("reports/tree_depth_experiment.png", dpi=300)

#analyze_tree(X, y)

def visualize_tree() -> None:
    tree = build_models(tree_depth=3, random_state=42).get("decision_tree")
    tree.fit(X, y)
    plt.figure(figsize=(16,10))

    plot_tree(
        tree,
        feature_names=X.columns,
        filled=True,
        rounded=True,
        impurity=False,
    )

    plt.savefig("reports/tree_visualization_depth_3.png", dpi=300)

#visualize_tree()

def analyze_forest(X : pd.DataFrame, y : pd.Series) -> None:
    forest = build_models(random_state=42).get("random_forest")
    forest.fit(X, y)
    importances = forest.feature_importances_
    feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    feature_importance.plot(kind="bar")
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("reports/forest_feature_importance.png", dpi=300)
    plt.show()

analyze_forest(X, y)