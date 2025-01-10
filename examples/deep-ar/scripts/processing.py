import pandas as pd


# Example processing script
def main():
    # Load your input data
    input_path = "/opt/ml/processing/input/AAPL.csv"
    df = pd.read_csv(input_path)

    # Perform some processing
    processed_data = df[df["Close"] > 100]

    # Save processed output
    output_path = "/opt/ml/processing/output/processed_data.csv"
    processed_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
