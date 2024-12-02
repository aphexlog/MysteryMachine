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

# Clean whitespace from all string columns
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Extract datetime features
data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
data["hour"] = data["datetime"].dt.hour
data["day_of_week"] = data["datetime"].dt.dayofweek + 1
data["year"] = data["datetime"].dt.year

# Encode categorical columns
label_encoder = LabelEncoder()
data["shape_encoded"] = label_encoder.fit_transform(data["shape"].fillna("unknown"))

# Convert mixed-type columns to numeric, setting invalid values to NaN
for col in ["duration (seconds)", "latitude", "longitude "]:  # Note the space after 'longitude'
    # First clean any whitespace
    if data[col].dtype == "object":
        data[col] = data[col].str.strip()
    # Then convert to numeric
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Drop rows with missing or invalid data in critical columns
data = data.dropna(subset=["latitude", "longitude", "duration (seconds)"])

# Scale numerical features
scaler = StandardScaler()
scaled_columns = [
    "duration (seconds)",
    "latitude",
    "longitude ",  # Note the space after 'longitude'
    "hour",
    "day_of_week",
    "year",
]
data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

# Prepare features for clustering
features = data[
    [
        "duration (seconds)",
        "latitude",
        "longitude",
        "hour",
        "day_of_week",
        "shape_encoded",
    ]
]

# with open("data.csv", "w") as f:
#     f.write(data.to_csv(index=False))


# # Apply K-means
# kmeans = KMeans(n_clusters=5, random_state=42)
# kmeans.fit(features)

# # Assign clusters to the data
# data["cluster"] = kmeans.labels_

# # Display clustered data
# print(data[["datetime", "city", "state", "cluster"]].head())
