"""
ML Engine for Housing Price Prediction
This module contains functions for training a machine learning model to predict housing prices based on the California housing dataset, and for making predictions using the trained model.
The `train_model` function trains a regression model (either Decision Tree or Random Forest based on cross-validation performance) and saves it to disk.
The `predict` function loads a trained model, takes input data in JSON or CSV format, preprocesses it, makes predictions, and saves the results to a specified output file in JSON or CSV format.
"""
import json
import logging
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle as pkl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def add_features(X : pd.DataFrame) -> pd.DataFrame:
    X["households"] = X["Population"] / X["AveOccup"]
    X["people_per_bedroom"] = X["AveOccup"] / X["AveBedrms"]
    X["bedrooms_per_room"] = X["AveBedrms"] / X["AveRooms"]
    return X

def load_dataset(data_csv : str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_csv)
    missing_rows = df.isna().any(axis=1).sum()
    if missing_rows > 0:
        logging.warning(f"Warning: {missing_rows} row/s with missing values will be dropped.")
    df = df.dropna()
    if df.empty:
        raise ValueError("The dataset is empty after dropping rows with missing values.")
    if "MedHouseVal" not in df.columns:
        raise ValueError("Target variable 'MedHouseVal' not found in the dataset.")
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    return X, y

def build_models(random_state: int = 42, tree_depth : int=5) -> dict:
    model_tree = DecisionTreeRegressor(
        max_depth=tree_depth,
        random_state=random_state
    )

    model_forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )

    return {
        "decision_tree": model_tree,
        "random_forest": model_forest,
    }

def train_model(data_csv : str, model_path : str = "model.pkl") -> dict:
    if not model_path:
        model_path = "model.pkl"
    if not data_csv.endswith(".csv"):
        raise ValueError("Invalid data file format. Please provide a CSV file.")
    if not model_path.endswith(".pkl"):
        raise ValueError("Invalid model file format. Please provide a .pkl file.")
    
    X, y = load_dataset(data_csv)
    #X = add_features(X)
    model_tree, model_forest = build_models().values()

    scores_tree = cross_val_score(model_tree, X, y, cv=5, scoring="r2")
    scores_forest = cross_val_score(model_forest, X, y, cv=5, scoring="r2")
    mean_tree = scores_tree.mean()
    std_tree = scores_tree.std()
    mean_forest = scores_forest.mean()
    std_forest = scores_forest.std()

    improvement = mean_forest - mean_tree
    threshold = 0.5 * (std_tree / (5 ** 0.5) + std_forest / (5 ** 0.5))
    if improvement < -threshold:
        selection = "tree"
        model = model_tree
        
    elif improvement > threshold:
        selection = "forest"
        model = model_forest
    else:
        selection = "close"
        model = model_tree

    model.fit(X, y)

    tree = {
        "mean" : mean_tree,
        "std" : std_tree
    }
    forest = {
        "mean" : mean_forest,
        "std" : std_forest,
    }
    result = {
        "selection" : selection,
        "model" : model,
        "tree" : tree,
        "forest" : forest,
        "data" : data_csv,
        "model_path" : model_path,
        "features" : X.columns.tolist()
    }

    with open(model_path, "wb") as f:
        pkl.dump(result, f)

    return result

def predict(input_data : str, model_path : str = None, output_path : str = None) -> str:
    if not model_path:
        model_path = "model.pkl"
    if not (input_data.endswith(".json") or input_data.endswith(".csv")):
        raise ValueError("Invalid prediction input file format. Please provide a JSON or CSV file.")
    if not model_path.endswith(".pkl"):
        raise ValueError("Invalid model file format. Please provide a .pkl file.")
    try:
        with open(model_path, "rb") as f:
            model_data = pkl.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if input_data.endswith(".csv"):
        df = pd.read_csv(input_data)
        df = add_features(df)
    else:
        with open(input_data, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Invalid JSON format. Please provide a JSON object or an array of JSON objects.")
        df = add_features(df)

    missing_rows = df.isna().any(axis=1).sum()
    if missing_rows > 0:
        logging.warning(f"Warning: {missing_rows} row/s with missing values will be dropped.")
    df = df.dropna()

    expected = model_data["features"]
    missing = [feat for feat in expected if feat not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in the input data: {', '.join(missing)}")
    input_data = df[expected]
    prediction = model_data["model"].predict(input_data)

    if not output_path:
        output_path = "predictions.json"
    if not (output_path.endswith(".json") or output_path.endswith(".csv")):
        raise ValueError("Invalid output file format. Please provide a JSON or CSV file.")
    output_df = df.copy()
    output_df["PredictedMedHouseVal"] = prediction
    if output_path.endswith(".csv"):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_json(output_path, orient="records", indent=2)
        
    return output_path