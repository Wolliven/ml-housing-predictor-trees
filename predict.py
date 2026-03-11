"""
Predict script for making predictions using the trained model
This script loads a trained model from disk, takes input data in JSON or CSV format,
preprocesses the data, makes predictions using the loaded model, and saves the predictions to a specified output file in JSON or CSV format.
"""
import sys
import argparse

def main() -> None:
    from ml_engine import predict
    parser = argparse.ArgumentParser(
        description="Generate predictions using a trained model",
        epilog="python predict.py --model model.pkl input.json/csv --out output.json/csv"
    )
    parser.add_argument("--model", type=str, required=False, help="Path to model to be used in the prediction")
    parser.add_argument("--out", type=str, required=False, help="Path to save the prediction results (JSON or CSV format)")
    parser.add_argument("input_path", type=str, help="Path to data to be used in the prediction")

    args = parser.parse_args()

    output_path = predict(args.input_path, model_path=args.model, output_path=args.out)
    print(f"Prediction completed. Results saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user. Exiting.")
        sys.exit(130)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)