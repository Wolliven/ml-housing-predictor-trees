"""
Entrypoint for training the model.
This script loads a dataset from a CSV file, preprocesses the data,
trains a regression model, evaluates its performance, and saves
the trained model to disk.
"""
import sys
import argparse

def main() -> None:
    from ml_engine import train_model
    parser = argparse.ArgumentParser(
        description="Train a housing price prediction model.",
        epilog="Example: python train.py data.csv --model_path model.pkl"
    )
    parser.add_argument("data_csv", type=str, help="Path to CSV dataset")
    parser.add_argument("--model_path", type=str, required=False, help="Path where the trained model will be saved")
    args = parser.parse_args()

    result = train_model(args.data_csv, model_path=args.model_path)
    print(f"Decision Tree Results:")
    print(f"Mean R² score: {result['tree']['mean']:.4f}±{result['tree']['std']:.4f}")
    print(f"Random Forest Results:")
    print(f"Mean R² score: {result['forest']['mean']:.4f}±{result['forest']['std']:.4f}")
    if result['selection'] == "close":
        print("The cross-validation results are very close, so the simpler model (Decision Tree) will be selected.")
    elif result['selection'] == "forest":
        print("Random Forest was selected based on cross-validation performance.")
    else:
        print("Decision Tree was selected based on cross-validation performance.")
    print(f"Training model with the following dataset: {result['data']}")
    print(f"Model saved to {result['model_path']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting.")
        sys.exit(130)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)