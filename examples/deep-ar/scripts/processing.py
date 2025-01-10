import os
import pandas as pd

# Get list of files in the input directory
input_files = os.listdir("/opt/ml/processing/input")
if not input_files:
    raise ValueError("No input files found")

# Read the first input file (assuming single file)
input_data_path = os.path.join("/opt/ml/processing/input", input_files[0])
output_data_path = "/opt/ml/processing/output/processed_train.csv"

# Read input data
df = pd.read_csv(input_data_path)

# Example processing: drop missing values
df_cleaned = df.dropna()
# df_cleaned = df_cleaned.drop(columns=["date"])

# Save processed data
os.makedirs("/opt/ml/processing/output", exist_ok=True)
df_cleaned.to_csv(output_data_path, index=False)
