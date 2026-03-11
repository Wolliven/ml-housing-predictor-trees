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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting.")
        sys.exit(130)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)