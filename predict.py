from src.prediction import predict
import argparse

def main():
    parser = argparse.ArgumentParser(description="SLP-for-Bio Predictor")
    parser.add_argument("--weight", "-w", type = str, required=True, help = "model weight file path")
    parser.add_argument("--seq", "-s", type = str, required=True, help = "sequence to predict")
    parser.add_argument("--type", "-t", type=str, default = "PROTEIN", help="(DNA/RNA/PROTEIN)")

    args = parser.parse_args()

    y = predict(args.weight, args.seq, args.type)
    print("Predicted Phenotype:", float(y))

if __name__ == "__main__":
    main()
