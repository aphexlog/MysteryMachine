import pandas as pd


def process_data(csv_in: str, csv_out: str) -> None:
    """
    Process the data in a CSV file.
        :param csv_in: The input CSV file.
        :param csv_out: The output CSV file.

    The function reads a CSV file, extracts the "id" and "destination_ip" columns,
    and writes the result to another CSV file.
    """
    df = pd.read_csv(csv_in)
    processed_df = df[["id", "destination_ip"]]
    processed_df.to_csv(csv_out, index=False, header=False)


if __name__ == "__main__":
    process_data("training.csv", "processed_training.csv")
    process_data("validation.csv", "processed_validation.csv")
