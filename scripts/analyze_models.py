from numpy import sqrt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
import pickle as pkl
from ml_engine import load_dataset, build_models, add_features

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
analyze_models(X, y, model_tree, model_forest)

#Features
X = add_features(X)

