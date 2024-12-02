import kagglehub
import pathlib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.cluster import KMeans

# Download latest version
path = kagglehub.dataset_download("NUFORC/ufo-sightings")

print("Path to dataset files:", path)

# Load data with low_memory=False to handle mixed types
data = pd.read_csv(pathlib.Path(path) / "scrubbed.csv", low_memory=False)

# Print column names to debug
print("\nColumn names in dataset:")
print(data.columns.tolist())

# Drop unnecessary columns
data = data.drop([
    "comments", 
    "duration (hours/min)", 
    "date posted",
    # Geographic columns - comment out these lines to include location data
    "country",
    "state",
    "city",
    # Shape column - comment out this line to include shape data
    "shape"
], axis=1)

# Clean whitespace from all string columns
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Extract datetime features
data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
# Extract features first
data["hour"] = data["datetime"].dt.hour
data["day_of_week"] = data["datetime"].dt.dayofweek + 1
data["year"] = data["datetime"].dt.year
# Then convert datetime to Unix timestamp (seconds since epoch)
data["datetime"] = data["datetime"].astype('int64') // 10**9


# Convert mixed-type columns to numeric, setting invalid values to NaN
for col in [
    "duration (seconds)",
    "latitude",
    "longitude ",
]:  # Note the space after 'longitude'
    # First clean any whitespace
    if data[col].dtype == "object":
        data[col] = data[col].str.strip()
    # Then convert to numeric
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Drop rows with missing or invalid data in critical columns
data = data.dropna(subset=["latitude", "longitude ", "duration (seconds)"])

# Scale numerical features
scaler = StandardScaler()
scaled_columns = [
    "duration (seconds)",
    "latitude",
    "longitude ",  # Note the space after 'longitude'
    "datetime",
    "hour",
    "day_of_week",
    "year",
]
data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

# Prepare features for clustering
features = [
    "duration (seconds)",
    "latitude",
    "longitude ",
    "hour",
    "day_of_week"
]

# Select only the features we want and drop any rows with NaN values
clean_data = data[features].dropna()

with open("training.csv", "w") as f:
    f.write(clean_data.to_csv(index=False, header=False))


# # Apply K-means
# kmeans = KMeans(n_clusters=5, random_state=42)
# kmeans.fit(features)

# # Assign clusters to the data
# data["cluster"] = kmeans.labels_

# # Display clustered data
# print(data[["datetime", "city", "state", "cluster"]].head())
