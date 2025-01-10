import os
import pandas as pd

# Simple data processing example
input_data_path = "/opt/ml/processing/input/train.csv"
output_data_path = "/opt/ml/processing/output/processed_train.csv"

# Read input data
df = pd.read_csv(input_data_path)

# Example processing: drop missing values
df_cleaned = df.dropna()
# df_cleaned = df_cleaned.drop(columns=["date"])

# Save processed data
os.makedirs("/opt/ml/processing/output", exist_ok=True)
df_cleaned.to_csv(output_data_path, index=False)
