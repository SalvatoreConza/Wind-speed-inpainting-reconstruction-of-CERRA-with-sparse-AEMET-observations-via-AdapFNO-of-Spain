# this script import the normalization method and saving method from the normalizion.py in
#  models folder I then call the normalization method that I wanted as ex "z_score"
import xarray as xr
import numpy as np
import yaml
from models.normalization import compute_stats, save_stats

# 1. Load Configuration to get file paths
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

train_path = config['dataset']['train_gt']  # Path to CERRA Training Data
vars_to_norm = ["ws"]  # The variable you want to normalize

print(f"Loading training data from {train_path}...")
ds_train = xr.open_dataset(train_path)

# 2. Compute Stats (Use 'z_score' for standard normalization)
print("Computing statistics (this may take a moment)...")
# Note: Ensure your ds has 'y' and 'x' dims as per your comment, 
# or rename them if they are 'latitude'/'longitude'
if 'latitude' in ds_train.dims:
    ds_train = ds_train.rename({'latitude': 'y', 'longitude': 'x'})

stats = compute_stats(ds_train, variables=vars_to_norm, method="z_score")

# 3. Save to disk
output_path = "data/stats.npz"
save_stats(stats, output_path)
print(f"Statistics saved to {output_path}")
print(f"Mean: {stats['mean']}, Std: {stats['std']}")