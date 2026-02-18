
import os
import random
import zlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.feature import canny
from pathlib import Path

def load_grayscale_image(path):
    img = Image.open(path).convert('L')  # ensure grayscale
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to [0,1]
    return arr


def show_random_images(dataset_key, datasets_dict):
    dataset = datasets_dict.get(dataset_key)
    if dataset is None:
        print(f"Dataset key '{dataset_key}' not found.")
        return
    
    df = dataset.get("df_raw")
    image_folder = dataset.get("image_path")
    dataset_name = dataset.get("name", dataset_key)

    if df is None or df.empty:
        print(f"No data in {dataset_name} dataframe to sample images.")
        return
    if not os.path.isdir(image_folder):
        print(f"Error: Image folder for {dataset_name} not found at {image_folder}")
        return
    
    print(f"\nRandom samples from {dataset_name} dataset:")
    sample_files = df["png_filename"].dropna().unique().tolist()
    if len(sample_files) < 3:
        print(f"Not enough images ({len(sample_files)}) in {dataset_name} to sample 3.")
        return
    
    random_samples = random.sample(sample_files, 3)
    
    plt.figure(figsize=(9, 3))
    for idx, fname in enumerate(random_samples):
        img_path = os.path.join(image_folder, fname)
        try:
            img = Image.open(img_path)
            plt.subplot(1, 3, idx + 1)
            plt.imshow(img, cmap='gray')
            short_fname = fname[:39]
            plt.title(f"{dataset_name}: {short_fname}", fontsize=10)
            plt.axis('off')
        except FileNotFoundError:
            print(f"File not found: {img_path}")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# Apply features to all datasets
# ---------------------------------------------


def apply_feature_to_datasets(feature_key: str, datasets_dict: dict, features: dict) -> None:
    """
    Calculate a single feature for every image in each dataset and add as a new column.
    Overrides existing values for that feature.

    Args:
      feature_key: Key in the `features` dict.
      datasets_dict: Dict of datasets (must have 'df_raw', 'df_features', 'image_path').
    """
    if feature_key not in features:
        raise KeyError(f"Feature '{feature_key}' not found in registry")

    feature_info = features[feature_key]
    feature_fn = feature_info["feature_fn"]
    print(f"[Feature Apply] Applying feature '{feature_info['name']}' to all datasets")

    for ds_key, ds in datasets_dict.items():
        df_raw = ds["df_raw"]
        df_feat = ds["df_features"]
        img_folder = ds["image_path"]
        print(f"  - Dataset '{ds_key}': {len(df_raw)} images")

        # Prepare column
        df_feat[feature_key] = np.nan

        # Compute feature per image
        for idx, row in df_raw.iterrows():
            fname = row.get("png_filename")
            if not isinstance(fname, str) or not fname:
                print(f"    [Warning] Missing filename at row {idx}")
                continue

            img_path = os.path.join(img_folder, fname)
            try:
                img_arr = load_grayscale_image(img_path)
                value = feature_fn(img_arr, ds_key)
                df_feat.at[idx, feature_key] = value
            except Exception as e:
                print(f"    [Error] Processing {img_path}: {e}")

        print(f"  -> Completed '{feature_info['name']}' for '{ds_key}'")

    print(f"[Feature Apply] All datasets updated with feature '{feature_key}'")



### new version applies to single dataset with normalization option
def apply_feature_to_single_dataset(
    feature_key: str,
    ds_key: str,
    ds: dict,
    features: dict,
    *,
    scale_value: float | None = None,
    quantile: float = 0.95
) -> float:
    """
    Compute feature for one dataset and write into ds['df_features'][feature_key].

    Normalization:
      - If scale_value is None: compute scale_value as the `quantile` of computed values
        within this dataset, then divide all values by it.
      - If scale_value is provided: do not recompute; just divide by it.

    Returns:
      scale_value used.
    """
    if feature_key not in features:
        raise KeyError(f"Feature '{feature_key}' not found in registry")

    feature_info = features[feature_key]
    feature_fn = feature_info["feature_fn"]

    df_raw = ds["df_raw"]
    df_feat = ds["df_features"]
    img_folder = ds["image_path"]

    print(f"  - Dataset '{ds_key}': {len(df_raw)} images")

    # Overwrite/prepare column
    df_feat[feature_key] = np.nan

    # Compute raw values once
    idx_vals: list[tuple[int, float]] = []
    vals: list[float] = []

    for idx, row in df_raw.iterrows():
        fname = row.get("png_filename")
        if not isinstance(fname, str) or not fname:
            print(f"    [Warning] Missing filename at row {idx}")
            continue

        img_path = os.path.join(img_folder, fname)
        try:
            img_arr = load_grayscale_image(img_path)
            v = feature_fn(img_arr, ds_key)
            if v is None:
                continue
            v = float(v)
            if not np.isfinite(v):
                continue
            idx_vals.append((idx, v))
            vals.append(v)
        except Exception as e:
            print(f"    [Error] Processing {img_path}: {e}")

    # Determine scaling constant
    if scale_value is None:
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            scale_value = 1.0
        else:
            scale_value = float(np.nanquantile(arr, quantile))
            if not np.isfinite(scale_value) or scale_value <= 0:
                raise ValueError(f"Computed scale_value is invalid: {scale_value}")
        print(f"    -> Computed scale_value (Q{quantile:.2f}) = {scale_value:.6g}")
    else:
        scale_value = float(scale_value)
        if not np.isfinite(scale_value) or scale_value <= 0:
            raise ValueError(f"Provided scale_value must be finite and > 0, got {scale_value}")
        print(f"    -> Using provided scale_value = {scale_value:.6g}")

    # Write normalized values
    for idx, raw_v in idx_vals:
        df_feat.at[idx, feature_key] = raw_v / scale_value

    print(f"  -> Completed normalized '{feature_info['name']}' for '{ds_key}'")
    return scale_value