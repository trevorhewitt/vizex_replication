

from scipy.ndimage import gaussian_laplace
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

# ---------------------------------------------
# Preprocessing functions for detail measure
# ---------------------------------------------

def preprocess_drawings_detail_old(image_array: np.ndarray, crop_border: int = 50, blur_radius: float = 5.0) -> np.ndarray:
    """
    Preprocess a 'drawings' image for detail analysis:
      1. Apply Gaussian blur with specified radius.
      2. Crop a fixed border from each side.
    """
    #print(f"[Drawings Preprocess] Original image shape: {image_array.shape}")
    # 1) Gaussian blur
    #blurred = gaussian_filter(image_array, sigma=blur_radius)
    ## disabling blur
    blurred = image_array

    #print(f"[Drawings Preprocess] Applied Gaussian blur with sigma={blur_radius}")

    # 2) Crop borders
    h, w = blurred.shape
    cropped = blurred[crop_border:h-crop_border, crop_border:w-crop_border]
    #print(f"[Drawings Preprocess] Cropped {crop_border}px border -> new shape: {cropped.shape}")

    return cropped

def preprocess_drawings_detail(
        image_array: np.ndarray,
        crop_border: int = 50,
        blur_radius: float = 5.0 # should always be 5
    ) -> np.ndarray:
        """
        Preprocess a 'drawings' image for detail analysis:
        1) Convert to grayscale if needed.
        2) Convert to float32.
        3) Gaussian blur.
        4) Crop fixed border.
        5) Normalize to [0,1] (global, per image).
        """
        arr = image_array

        # Grayscale if RGB/RGBA
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            rgb = arr[..., :3].astype(np.float32)
            arr = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

        arr = arr.astype(np.float32, copy=False)
        blurred = gaussian_filter(arr, sigma=blur_radius)

        h, w = blurred.shape[:2]
        if crop_border > 0:
            if 2 * crop_border >= h or 2 * crop_border >= w:
                raise ValueError(f"crop_border={crop_border} too large for image shape {blurred.shape}")
            blurred = blurred[crop_border:h - crop_border, crop_border:w - crop_border]

        mn = float(blurred.min())
        ptp = float(np.ptp(blurred))
        norm = (blurred - mn) / (ptp + 1e-8)

        return norm


def preprocess_interface_detail(image_array: np.ndarray, hp_sigma: float = 10.0, blur_radius: float = 2.0) -> np.ndarray:
    """
    Preprocess an 'interface' image for detail analysis:
      1. Apply a high-pass filter (subtract a broad Gaussian blur) to remove low-frequency vignetting.
      2. Normalize contrast to span [0,1].
      3. Apply Gaussian blur with specified radius.
    """
    #print(f"[Interface Preprocess] Original image shape: {image_array.shape}")
    # 1) High-pass filter: subtract a broad low-pass (Gaussian) version
    ## DISABLED
    low_pass = gaussian_filter(image_array, sigma=hp_sigma)
    high_pass = image_array - low_pass
    #high_pass = image_array
    #print(f"[Interface Preprocess] Applied high-pass filter with sigma={hp_sigma}")

    # 2) Contrast normalization to [0,1]
    min_val, max_val = high_pass.min(), high_pass.max()
    norm = (high_pass - min_val) / (max_val - min_val + 1e-8)
    #print(f"[Interface Preprocess] Normalized contrast (min={min_val:.4f}, max={max_val:.4f})")

    # 3) Gaussian blur
    # DISABLED
    blurred = norm #gaussian_filter(norm, sigma=blur_radius)
    #print(f"[Interface Preprocess] Applied Gaussian blur with sigma={blur_radius}")

    return blurred


def preprocess_for_detail(image_array: np.ndarray, dataset_key: str) -> np.ndarray:
    """
    Dispatch preprocessing based on dataset key.
    """
    if dataset_key.lower() == "drawings" or dataset_key.lower() == "drawings_targets":
        return preprocess_drawings_detail(image_array)
    elif dataset_key.lower() == "interface" or dataset_key.lower() == "interface_targets":
        return preprocess_interface_detail(image_array)
    else:
        raise ValueError(f"Unknown dataset key for detail preprocessing: {dataset_key}")

def calculate_detail(
    image_array: np.ndarray,
    dataset_key: str,
    log_sigma: float = 1.0, # included for compatibility but it should always be 1.0
    unpreprocessed: bool = False
) -> float:
    """
    Detail via LoG, with magnitude preserved across patches.

    Contract:
      - If unpreprocessed=True: we preprocess (which includes GLOBAL normalization).
      - If unpreprocessed=False: we assume input is already preprocessed + globally normalized.
        We do NOT renormalize per patch.

    Returns a value in [0,1] via an optional global scale factor.
    """
    if image_array is None or (isinstance(image_array, np.ndarray) and image_array.size == 0):
        return 0.0

    proc = image_array
    if unpreprocessed:
        proc = preprocess_for_detail(proc, dataset_key)

    proc = proc.astype(np.float32, copy=False)

    log_edges = np.abs(gaussian_laplace(proc, sigma=log_sigma))
    edge_strength = float(np.mean(log_edges))

    # 4) Scaling factors
    #multiplier = 0
    #if dataset_key == "drawings":
    #    multiplier = 92.5
    #elif dataset_key == "interface":
    #    multiplier = 6.6
    #else:
    #    raise ValueError(f"Unknown dataset key for detail calculation: {dataset_key}")
    
    return edge_strength #* multiplier

def calculate_detail_old(image_array: np.ndarray, 
                     dataset_key: str,
                     log_sigma: float = 1.0,
                     unpreprocessed: bool = True) -> float:
    """
    Calculate the 'detail' of an image using Laplacian of Gaussian:
      1. Preprocess image.
      2. Apply LoG to detect edge-rich regions.
      3. Return average edge strength (normalized to [0,1]).
    """
    if image_array is None or (isinstance(image_array, np.ndarray) and image_array.size == 0):
        return 0

    # 1) Preprocess
    proc = image_array
    if unpreprocessed:
        proc = preprocess_for_detail(image_array, dataset_key)

    # Normalize image to [0,1] before LoG
    proc = (proc - proc.min()) / (np.ptp(proc) + 1e-6)

    # 2) LoG edge magnitude (absolute to ensure positivity)
    log_edges = np.abs(gaussian_laplace(proc, sigma=log_sigma))

    # 3) Normalize: scale edge map to [0,1] for consistency across images
    max_val = log_edges.max()
    if max_val > 0:
        norm_log = log_edges / max_val
    else:
        norm_log = log_edges  # all zeros

    edge_fraction = float(norm_log.mean())

    # 4) Scaling factors
    multiplier = 0
    if dataset_key == "drawings":
        multiplier = 92.5
    elif dataset_key == "interface":
        multiplier = 3.3
    else:
        raise ValueError(f"Unknown dataset key for detail calculation: {dataset_key}")


    # 5) Clip and return
    detail_value = np.clip(edge_fraction, 0.0, 1.0) * multiplier

    return detail_value



# 1) New “complexity” feature function with optional preprocessing
def calculate_complexity(image_array: np.ndarray,
                         dataset_key: str,
                         unpreprocessed: bool = True) -> float:
    """
    Zip‐complexity ratio of an image:
      1) If unpreprocessed: run the same preprocessing as for detail
      2) Compute bytes, compress, and return (compressed / raw) in [0,1]
    """
    # 1) optional preprocessing
    if unpreprocessed:
        image_array = preprocess_for_detail(image_array, dataset_key)

    # 2) compute raw vs compressed bytes
    raw_bytes = (image_array * 255).astype(np.uint8).tobytes()
    comp_bytes = zlib.compress(raw_bytes)
    ratio = len(comp_bytes) / len(raw_bytes)

    # 3) clip and return
    return float(np.clip(ratio, 0.0, 1.0))



    ####### ORDER AND CONTENT FEATURE


# ------------------------------------------------------------------
# 1) Adjustable stretch parameter for the “order” feature
# ------------------------------------------------------------------
# Tweak this to increase/decrease the range along the orthogonal diagonal
stretch = 2.0


# ------------------------------------------------------------------
# 2) Feature functions for Content Level & Order
# ------------------------------------------------------------------
def calculate_content(image_array: np.ndarray,
                      dataset_key: str,
                      unpreprocessed: bool = True) -> float:
    """
    Content Level = average of Detail and Complexity.
    Requires those features to be registered.
    """
    # compute sub‐features
    d = calculate_detail(image_array, dataset_key, unpreprocessed=unpreprocessed)
    c = calculate_complexity(image_array, dataset_key, unpreprocessed=unpreprocessed)
    # average & clip to [0,1]
    return float(np.clip((d + c) / 2.0, 0.0, 1.0))


def calculate_order(image_array: np.ndarray,
                    dataset_key: str,
                    unpreprocessed: bool = True) -> float:
    """
    Order = projection of (complexity, detail) onto the orthogonal diagonal,
    normalized to [0,1], with adjustable stretch.
      - If `unpreprocessed=True`, this is a full‐image call: we preprocess,
        then compute detail & complexity on that full image.
      - If `unpreprocessed=False`, this is a patch call: we skip all
        cropping/gaussian steps and just run detail/complexity in-place
        on the patch, so we never get empty arrays.
    """
    # Full‐image path (normal behavior)
    if unpreprocessed:
        d = calculate_detail(image_array, dataset_key, unpreprocessed=True)
        c = calculate_complexity(image_array, dataset_key, unpreprocessed=True)
    else:
        # Patch path: force both detail & complexity to skip preprocessing
        d = calculate_detail(image_array, dataset_key, unpreprocessed=False)
        c = calculate_complexity(image_array, dataset_key, unpreprocessed=False)

    # linear mapping to [0,1]: (d - c) in [-1,1] → [0,1]c
    order = (d - c + 1.0) / 2.0

    # apply stretch around center 0.5, then clip
    if stretch != 1.0:
        order = (order - 0.5) * stretch + 0.5

    return float(np.clip(order, 0.0, 1.0))

