import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from PIL import Image
from pathlib import Path


project_path = Path().resolve().parent
tables_path = os.path.join(project_path, "015_tables")
raw_data_path = os.path.join(project_path, "000_raw_data")
raw_data_path_interface = os.path.join(raw_data_path, "interface")
raw_data_path_drawings = os.path.join(raw_data_path, "drawings")
viz_path = os.path.join(project_path, "020_visualizations")


def load_grayscale_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _apply_rescale(value, rescale: float = 1.0, clip: bool = True):
    """
    Linear rescaling used to bring raw feature outputs into the same scale
    as precomputed tables (typically [0, 1]). Backwards compatible by default.

    - If rescale==1, no change.
    - If clip==True, clips to [0, 1] after rescaling.
    """
    if rescale is None:
        rescale = 1.0
    v = np.asarray(value, dtype=np.float32) * float(rescale)
    if clip:
        v = np.clip(v, 0.0, 1.0)
    if np.isscalar(value):
        return float(v)
    return v


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
            plt.imshow(img, cmap="gray")
            short_fname = fname[:39]
            plt.title(f"{dataset_name}: {short_fname}", fontsize=10)
            plt.axis("off")
        except FileNotFoundError:
            print(f"File not found: {img_path}")
    plt.tight_layout()
    plt.show()


# 1) Function to render binned feature heatmap
def plot_binned_feature(
    image_array,
    feature_fn,
    bins: int = 7,
    print_values: bool = True,
    ax=None,
    rescale: float = 1.0,
    clip: bool = True,
):
    """
    IMPORTANT CONTRACT:
      - image_array must already be preprocessed appropriately for patching.
      - feature_fn MUST be called on patches with unpreprocessed=False.
      - rescale is applied to the per-patch feature outputs (default 1 = no change).
    """
    h, w = image_array.shape
    bin_h = h // bins
    bin_w = w // bins

    feature_map = np.zeros((bins, bins), dtype=np.float32)

    for i in range(bins):
        for j in range(bins):
            y0, y1 = i * bin_h, (i + 1) * bin_h if i < bins - 1 else h
            x0, x1 = j * bin_w, (j + 1) * bin_w if j < bins - 1 else w
            patch = image_array[y0:y1, x0:x1]

            # feature_fn must NOT preprocess again; enforce unpreprocessed=False
            raw_val = feature_fn(patch, unpreprocessed=False)
            feature_map[i, j] = _apply_rescale(raw_val, rescale=rescale, clip=clip)

    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(feature_map, cmap="rainbow", vmin=0, vmax=1, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])

    if print_values:
        for i in range(bins):
            for j in range(bins):
                val = feature_map[i, j]
                ax.text(j, i, f"{val:.2f}".lstrip("0"), ha="center", va="center", fontsize=11)

    return im, feature_map


# 2) Function to create horizontal colorbar legend
def make_color_legend(num_ticks: int = 5, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap="rainbow")
    sm.set_array([])
    cbar = plt.colorbar(
        sm,
        cax=ax,
        orientation="horizontal",
        ticks=np.linspace(0, 1, num_ticks),
        fraction=0.046,
        pad=0.02,
    )
    ax.set_xlabel("Feature value")
    ax.set_yticks([])
    return cbar


def plot_feature_histograms(
    feature_key: str,
    dataset_key: str,
    datasets_dict: dict,
    features_dict: dict,
    bins: int = 30,
    show_norm: bool = False,
    negative: bool = False,
) -> None:
    """
    Histogram uses PRECOMPUTED feature values from df_features.
    Those are assumed to already be scaled/normalized appropriately.
    """
    ds = datasets_dict.get(dataset_key)
    if ds is None:
        raise KeyError(f"Dataset '{dataset_key}' not found")
    df_feat = ds["df_features"]

    data = df_feat[feature_key].dropna().values
    if data.size == 0:
        print(f"No non-NaN values for '{feature_key}' in dataset '{dataset_key}'.")
        return

    data_min, data_max = float(np.min(data)), float(np.max(data))
    if data_min < 0:
        abs_max = max(abs(data_min), abs(data_max))
        xmin, xmax = -abs_max, abs_max
    else:
        xmin, xmax = 0, data_max

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, range=(xmin, xmax), edgecolor="black")
    plt.title(f"{ds['name']} - {feature_key} Distribution")
    plt.xlabel(feature_key)
    plt.ylabel("Frequency")
    plt.xlim(xmin, xmax)

    out_file = os.path.join(viz_path, f"hist_{feature_key}_{dataset_key}.png")
    plt.savefig(out_file, dpi=300)
    plt.show()
    
def visualize_spatial_feature_on_image(
    feature_key: str,
    dataset_key: str,
    datasets_dict: dict,
    features_dict: dict,
    image_idx: int = None,
    n_x_bins: int = 35,
    rescale: float = 1.0,
    preprocess_dataset_key: str | None = None,  # NEW
) -> None:
    """
    preprocess_dataset_key:
      - if None, uses dataset_key (backwards compatible)
      - if provided, is the key passed into preprocess_fn/feature_fn
        (useful when e.g. interface_targets should preprocess like 'interface')
    """
    if dataset_key not in datasets_dict:
        raise KeyError(f"Dataset '{dataset_key}' not found")
    if feature_key not in features_dict:
        raise KeyError(f"Feature '{feature_key}' not found")

    ds = datasets_dict[dataset_key]
    df_raw = ds["df_raw"]
    df_feat = ds["df_features"]
    img_folder = ds["image_path"]
    dataset_name = ds.get("name", dataset_key)

    feat_info = features_dict[feature_key]
    feature_fn = feat_info["feature_fn"]
    preprocess_fn = feat_info["preprocess_fn"]

    # NEW: key used for preprocessing/feature computation
    dk = preprocess_dataset_key if preprocess_dataset_key is not None else dataset_key

    valid_idxs = df_raw.index.tolist()
    if image_idx is None or image_idx not in valid_idxs:
        image_idx = random.choice(valid_idxs)

    fname = df_raw.loc[image_idx, "png_filename"]
    img_path = os.path.join(img_folder, fname)
    img_arr = load_grayscale_image(img_path)

    overall_val = df_feat.at[image_idx, feature_key]

    # preprocess ONCE using dk
    proc_arr = preprocess_fn(img_arr, dk)

    # patch feature wrapper: no preprocessing in patches; rescale; clip to [0,1] inside plot_binned_feature already,
    # but we already ensured there is no clipping at this level by not clipping here. (Your earlier function clipped;
    # keep that if desired. For this single-image viz, you probably still want [0,1] display.)
    def patch_feature_fn(patch, unpreprocessed=False):
        raw = feature_fn(patch, dk, unpreprocessed=unpreprocessed)
        return _apply_rescale(raw, rescale=rescale, clip=True)

    fig = plt.figure(figsize=(9, 4.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_arr, cmap="gray", vmin=0, vmax=1)
    ax0.set_title("Original Image")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(proc_arr, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Preprocessed Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    plot_binned_feature(proc_arr, patch_feature_fn, bins=n_x_bins, ax=ax2, print_values=False, rescale=1.0)
    ax2.set_title(f"{n_x_bins}×{n_x_bins} bins")

    ax3 = fig.add_subplot(gs[0, 3])
    plot_binned_feature(proc_arr, patch_feature_fn, bins=7, ax=ax3, print_values=True, rescale=1.0)
    ax3.set_title("7×7 bins")

    fig.suptitle(f"{dataset_name.capitalize()} Index {image_idx} – {feat_info['name']}", fontsize=16)
    fig.text(0.5, 0.91, f"Overall {feat_info['name']}: {overall_val:.3f}", ha="center", va="center", fontsize=14)

    ax_cb = fig.add_subplot(gs[1, :])
    make_color_legend(num_ticks=6, ax=ax_cb)
    ax_cb.plot(overall_val, 0.5, "o", transform=ax_cb.get_xaxis_transform(), markersize=10)

    fig.subplots_adjust(wspace=0.3, hspace=0.4, top=0.88, bottom=0.12)
    plt.tight_layout()
    save_fname = f"{feature_key}_{dataset_key}_idx{image_idx}.png"
    save_path = os.path.join(viz_path, save_fname)
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_spatial_feature_spectrum(
    feature_key: str,
    dataset_key: str,
    datasets_dict: dict,
    features_dict: dict,
    n: int = 7,
    seed: int = None,
    rescale: float = 1.0,
) -> None:
    if dataset_key not in datasets_dict:
        raise KeyError(f"Dataset '{dataset_key}' not found")
    if feature_key not in features_dict:
        raise KeyError(f"Feature '{feature_key}' not found")

    ds = datasets_dict[dataset_key]
    df_raw = ds["df_raw"]
    df_feat = ds["df_features"]
    img_folder = ds["image_path"]
    dataset_name = ds.get("name", dataset_key)

    feat_info = features_dict[feature_key]
    feature_fn = feat_info["feature_fn"]
    preprocess_fn = feat_info["preprocess_fn"]

    records = []
    for idx, row in df_raw.iterrows():
        fname = row.get("png_filename")
        if not isinstance(fname, str):
            continue
        val = df_feat.at[idx, feature_key]  # precomputed (already scaled)
        if pd.isna(val):
            continue
        records.append((fname, float(val)))

    if not records:
        print("No images to display.")
        return

    records.sort(key=lambda x: x[1])
    m = len(records)

    if m <= n:
        selected = records.copy()
    else:
        if seed is not None:
            random.seed(seed)

        values = np.array([val for (_, val) in records], dtype=np.float32)
        min_val, max_val = float(values[0]), float(values[-1])

        if n == 1:
            target_values = np.array([(min_val + max_val) / 2.0], dtype=np.float32)
        else:
            target_values = np.linspace(min_val, max_val, n, dtype=np.float32)

        used_indices = set()
        selected = []
        for t in target_values:
            diffs = np.abs(values - t)
            for ui in used_indices:
                diffs[ui] = np.inf
            best_idx = int(np.argmin(diffs))
            used_indices.add(best_idx)
            selected.append(records[best_idx])

    def patch_feature_fn(patch, unpreprocessed=False):
        raw = feature_fn(patch, dataset_key, unpreprocessed=unpreprocessed)
        return _apply_rescale(raw, rescale=rescale, clip=True)

    fig = plt.figure(figsize=(n * 2, 8))
    gs = fig.add_gridspec(4, n, height_ratios=[1, 1, 1, 1])

    for col, (fname, val) in enumerate(selected):
        img_path = os.path.join(img_folder, fname)
        arr = load_grayscale_image(img_path)
        proc = preprocess_fn(arr, dataset_key)

        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax0.set_title(f"{val:.2f}", fontsize=16)
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(proc, cmap="gray", vmin=0, vmax=1)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[2, col])
        plot_binned_feature(proc, patch_feature_fn, bins=35, print_values=False, ax=ax2, rescale=1.0)
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[3, col])
        plot_binned_feature(proc, patch_feature_fn, bins=7, print_values=False, ax=ax3, rescale=1.0)
        ax3.axis("off")

    plt.suptitle(f"{dataset_name.capitalize()} {feat_info['name']} Spectrum (n={n})", fontsize=14)
    plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.92, bottom=0.05)

    fname_out = f"spectrum_{feature_key}_{dataset_key}_n{n}.png"
    path_out = os.path.join(viz_path, fname_out)
    plt.savefig(path_out, dpi=300)
    plt.show()


def full_spatial_feature_viz(
    feature_key: str,
    dataset_key: str,
    datasets_dict: dict,
    features_dict: dict,
    rescale: float = 1.0,
    **viz_kwargs,
) -> None:
    """
    Note:
      - Summary stats + histogram are from precomputed values (already scaled).
      - Spatial heatmaps computed on-the-fly apply `rescale` (default 1).
    """
    ds = datasets_dict.get(dataset_key)
    if ds is None:
        raise KeyError(f"Dataset '{dataset_key}' not found")
    df_feat = ds["df_features"]

    data = df_feat[feature_key].dropna().values
    if data.size == 0:
        print(f"No non-NaN values for '{feature_key}' in dataset '{dataset_key}'.")
        return

    min_val = float(np.min(data))
    max_val = float(np.max(data))
    mean_val = float(np.mean(data))
    sd_val = float(np.std(data, ddof=1))

    print(
        f"{feature_key} | "
        f"min={min_val:.4f}, "
        f"max={max_val:.4f}, "
        f"mean={mean_val:.4f}, "
        f"sd={sd_val:.4f}"
    )

    plot_feature_histograms(
        feature_key,
        dataset_key,
        datasets_dict,
        features_dict,
        **{k: viz_kwargs[k] for k in ["bins", "show_norm"] if k in viz_kwargs},
    )

    for _ in range(3):
        visualize_spatial_feature_on_image(
            feature_key,
            dataset_key,
            datasets_dict,
            features_dict,
            rescale=rescale,
            **{k: viz_kwargs[k] for k in ["image_idx", "n_x_bins"] if k in viz_kwargs},
        )

    plot_spatial_feature_spectrum(
        feature_key,
        dataset_key,
        datasets_dict,
        features_dict,
        rescale=rescale,
        **{k: viz_kwargs[k] for k in ["n", "seed"] if k in viz_kwargs},
    )


def plot_2d_frequency_heatmap(
    feature_key_x: str,
    feature_key_y: str,
    dataset_key: str,
    datasets_dict: dict,
    features_dict: dict,  # kept for signature compatibility
    bins_x: int = 10,
    bins_y: int = 10,
    negative: bool = False,
) -> None:
    """
    Uses PRECOMPUTED values (already scaled). No rescale needed here.
    """
    minv = -1 if negative else 0

    ds = datasets_dict[dataset_key]
    df_feat = ds["df_features"]
    dataset_name = ds.get("name", dataset_key)

    x_vals = df_feat[feature_key_x].dropna().values
    y_vals = df_feat[feature_key_y].dropna().values
    if x_vals.size == 0 or y_vals.size == 0:
        print("No data to display.")
        return

    counts, x_edges, y_edges = np.histogram2d(
        x_vals,
        y_vals,
        bins=[bins_x, bins_y],
        range=[[minv, 1], [minv, 1]],
    )
    counts = counts.T
    masked = np.ma.masked_where(counts == 0, counts)

    cmap = plt.cm.rainbow.copy()
    cmap.set_bad(color="white")

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        masked,
        origin="lower",
        cmap=cmap,
        extent=[minv, 1, minv, 1],
        aspect="auto",
        vmin=1,
        vmax=masked.max(),
    )
    plt.colorbar(im, label="Frequency")
    plt.xlabel(feature_key_x)
    plt.ylabel(feature_key_y)
    plt.title(f"{dataset_name} 2D Frequency: {feature_key_x} vs {feature_key_y}")

    fname = f"freq_heatmap_{feature_key_x}_{feature_key_y}_{dataset_key}.png"
    path = os.path.join(viz_path, fname)
    plt.savefig(path, dpi=300)
    plt.show()


def plot_2d_spectrum(
    feature_key_x: str,
    feature_key_y: str,
    dataset_key: str,
    datasets_dict: dict,
    features_dict: dict,  # kept for signature compatibility
    n_x: int = 5,
    n_y: int = 5,
    seed: int = None,
) -> None:
    """
    Uses PRECOMPUTED values (already scaled). No rescale needed here.
    """
    ds = datasets_dict[dataset_key]
    df_raw = ds["df_raw"]
    df_feat = ds["df_features"]
    img_folder = ds["image_path"]
    dataset_name = ds.get("name", dataset_key)

    records = []
    for idx, row in df_raw.iterrows():
        fname = row.get("png_filename")
        val_x = df_feat.at[idx, feature_key_x]
        val_y = df_feat.at[idx, feature_key_y]
        if isinstance(fname, str) and not np.isnan(val_x) and not np.isnan(val_y):
            records.append((fname, float(val_x), float(val_y)))

    if not records:
        print("No images to display.")
        return

    fnames, xs, ys = zip(*records)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    grid_x = np.linspace(min_x, max_x, n_x, dtype=np.float32)
    grid_y = np.linspace(min_y, max_y, n_y, dtype=np.float32)

    dx = (max_x - min_x) / (n_x - 1) if n_x > 1 else 1.0
    dy = (max_y - min_y) / (n_y - 1) if n_y > 1 else 1.0
    tol = float(np.hypot(dx, dy))

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    fig, ax = plt.subplots(figsize=(n_x, n_y))
    ax.set_facecolor("#F0F0F0")
    ax.set_xlim(min_x - dx / 2, max_x + dx / 2)
    ax.set_ylim(min_y - dy / 2, max_y + dy / 2)
    ax.set_xticks(grid_x)
    ax.set_yticks(grid_y)
    ax.set_xlabel(feature_key_x)
    ax.set_ylabel(feature_key_y)
    ax.set_title(f"{dataset_name} {feature_key_x} vs {feature_key_y} Spectrum")
    for spine in ax.spines.values():
        spine.set_zorder(10)
    ax.set_axisbelow(False)

    pos = ax.get_position()
    w_frac = pos.width / n_x
    h_frac = pos.height / n_y

    used = set()
    for gx in grid_x:
        for gy in grid_y:
            dists = np.hypot(xs - gx, ys - gy)
            candidates = [i for i, d in enumerate(dists) if d <= tol and i not in used]
            if not candidates:
                continue
            idx = random.choice(candidates)
            used.add(idx)
            fname = fnames[idx]
            arr = load_grayscale_image(os.path.join(img_folder, fname))

            fx = pos.x0 + ((gx - (min_x - dx / 2)) / ((max_x + dx / 2) - (min_x - dx / 2))) * pos.width
            fy = pos.y0 + ((gy - (min_y - dy / 2)) / ((max_y + dy / 2) - (min_y - dy / 2))) * pos.height
            left = fx - w_frac / 2
            bottom = fy - h_frac / 2

            ax_img = fig.add_axes([left, bottom, w_frac, h_frac], frameon=False)
            ax_img.imshow(arr, cmap="gray", vmin=0, vmax=1, aspect="equal")
            ax_img.axis("off")

    fname = f"spectrum2d_{feature_key_x}_{feature_key_y}_{dataset_key}.png"
    path = os.path.join(viz_path, fname)
    plt.savefig(path, dpi=300)
    plt.show()


def plot_image_overview(
    filenames,
    filepath=None,
    axes=None,
    figsize=(4, 4),
    title=None,
    border_width=1,
):
    """
    Overview of a set of images (1, <=9 in a 3x3, or >=10 as 6 samples + "...").
    """
    num_images = len(filenames)

    if filepath:
        full_paths = [os.path.join(filepath, fname) for fname in filenames]
    else:
        full_paths = filenames.copy()

    randomized = random.sample(full_paths, num_images)

    def _imshow_with_border(ax, img_arr, linewidth):
        ax.imshow(img_arr, cmap="gray")
        ax.axis("off")
        rect = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            linewidth=linewidth,
            edgecolor="black",
            facecolor="none",
            zorder=10,
            clip_on=False,
        )
        ax.add_patch(rect)

    fig_title = f"n = {num_images} images" if title is None else f"{title} (n = {num_images} images)"

    if num_images == 1:
        if axes is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        else:
            ax = axes
            fig = ax.get_figure()

        img_arr = load_grayscale_image(randomized[0])
        _imshow_with_border(ax, img_arr, linewidth=border_width)
        fig.suptitle(fig_title, y=0.98, fontsize=12)

        if axes is None:
            plt.tight_layout()
            return fig, ax
        return ax

    if num_images <= 9:
        if axes is None:
            fig, axes_grid = plt.subplots(3, 3, figsize=figsize)
            axes_list = axes_grid.flatten()
        else:
            axes_list = list(axes)
            if len(axes_list) != 9:
                raise ValueError("When plotting ≤9 images, `axes` must have length 9.")
            try:
                axes_grid = axes.reshape(3, 3)
            except Exception:
                axes_grid = axes_list
            fig = axes_list[0].get_figure()

        for i in range(num_images):
            ax_i = axes_list[i]
            img_arr = load_grayscale_image(randomized[i])
            _imshow_with_border(ax_i, img_arr, linewidth=border_width)

        for j in range(num_images, 9):
            axes_list[j].axis("off")

        fig.suptitle(fig_title, y=0.98, fontsize=12)

        if axes is None:
            plt.tight_layout()
            return fig, axes_grid
        return axes_grid

    selected = randomized[:6]

    if axes is None:
        fig = plt.figure(figsize=figsize)
        img_axes = []
        box_w, box_h = 0.3, 0.3

        for i in range(3):
            left = 0.05 + i * 0.32
            bottom = 0.6 - i * 0.025
            ax_i = fig.add_axes([left, bottom, box_w, box_h])
            img_axes.append(ax_i)

        ax_text = fig.add_axes([0.35, 0.45, 0.3, 0.15])

        for i in range(3, 6):
            idx = i - 3
            left = 0.05 + idx * 0.32
            bottom = 0.15 - idx * 0.025
            ax_i = fig.add_axes([left, bottom, box_w, box_h])
            img_axes.append(ax_i)
    else:
        if len(axes) != 7:
            raise ValueError("For ≥10 images, `axes` must be a list of 7 Axes (6 for images + 1 for text).")
        img_axes = axes[:6]
        ax_text = axes[6]
        fig = img_axes[0].get_figure()

    for i, ax_img in enumerate(img_axes):
        img_arr = load_grayscale_image(selected[i])
        _imshow_with_border(ax_img, img_arr, linewidth=border_width)

    ax_text.axis("off")
    ax_text.text(0.5, 0.5, "...", fontsize=20, ha="center", va="center")

    fig.suptitle(fig_title, y=0.98, fontsize=12)

    if axes is None:
        plt.tight_layout()
        return fig, img_axes, ax_text
    return img_axes, ax_text


def make_condition_average_heatmaps_and_scale(
    feature_key: str,
    dataset_key: str,
    datasets_dict: dict,
    features_dict: dict,
    bins: int = 35,
    figsize=(3, 3),
    cmap_use: str = "inferno",
    rescale: float = 1.0,
    condition_map: dict | None = None,
    condition_col: str = "condition_code",
    include_conditions: tuple[str, ...] = ("1", "4", "8"),
    save_prefix: str | None = None,
    scale_save_name: str | None = None,
    scale_label: str | None = None,
    scale_width: float = 1.4,
    scale_height: float = 3.0,
    scale_text_size: float = 12,
    scale_border_thickness: float = 1.5,
    scale_tick_distance: float | None = None,
    show: bool = True,
):
    """
    Creates one averaged heatmap per condition (separate figures), plus ONE shared scale figure.

    Key requirements satisfied:
      - preprocess full image ONCE, then patch, then feature_fn(patch, unpreprocessed=False)
      - apply linear rescale to patch feature values (default 1.0 = backwards compatible)
      - NO clipping (values may exceed 1)
      - shared scale: vmin=0 and vmax=max(1, global_max_cell_across_all_condition_heatmaps)

    Saves tightly cropped PNGs to viz_path (same directory used elsewhere in viz.py).

    Returns
    -------
    heatmaps_by_condition : dict[str, np.ndarray]
    vmin : float
    vmax : float
    """

    # Defaults
    if condition_map is None:
        condition_map = {"1": "11 Hz", "4": "40 Hz", "8": "80 Hz"}

    # Validate
    if dataset_key not in datasets_dict:
        raise KeyError(f"Dataset '{dataset_key}' not found")
    if feature_key not in features_dict:
        raise KeyError(f"Feature '{feature_key}' not found")

    ds = datasets_dict[dataset_key]
    df_raw = ds.get("df_raw")
    img_folder = ds.get("image_path")
    dataset_name = ds.get("name", dataset_key)

    if df_raw is None or df_raw.empty:
        print(f"[make_condition_average_heatmaps_and_scale] No df_raw for {dataset_key}, skipping.")
        return {}, 0.0, 1.0
    if not isinstance(img_folder, str) or not os.path.isdir(img_folder):
        raise FileNotFoundError(f"Image folder not found for dataset '{dataset_key}': {img_folder}")

    feat_info = features_dict[feature_key]
    feature_fn = feat_info["feature_fn"]
    preprocess_fn = feat_info["preprocess_fn"]
    feat_label = scale_label if scale_label is not None else feat_info.get("name", feature_key)

    n_rows = int(bins)
    n_cols = int(bins)

    # Helper: safe-ish filename
    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in s)

    # Compute avg heatmap per condition (store first; plot after we know global vmax)
    heatmaps_by_condition: dict[str, np.ndarray] = {}
    global_max = 0.0

    for cond_code in include_conditions:
        cond_name = condition_map.get(str(cond_code), str(cond_code))

        # Filter df for this condition
        if condition_col not in df_raw.columns:
            raise KeyError(f"df_raw is missing condition column '{condition_col}'")
        df_cond = df_raw[df_raw[condition_col].astype(str) == str(cond_code)]
        num_images = len(df_cond)

        if num_images == 0:
            print(f"[{dataset_key}] No images for condition {cond_name} ({cond_code}), skipping.")
            continue

        if "png_filename" not in df_cond.columns:
            raise KeyError("df_raw is missing required column 'png_filename'")

        filenames = df_cond["png_filename"].dropna().astype(str).tolist()
        if len(filenames) == 0:
            print(f"[{dataset_key}] No valid filenames for condition {cond_name} ({cond_code}), skipping.")
            continue

        heatmap_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
        used = 0

        for fname in filenames:
            img_path = os.path.join(img_folder, fname)
            if not os.path.isfile(img_path):
                continue

            # load -> preprocess ONCE
            img = load_grayscale_image(img_path)  # [0,1] float32
            pre = preprocess_fn(img, dataset_key)
            h_pre, w_pre = pre.shape[:2]

            # patch feature WITHOUT preprocessing again
            for i in range(n_rows):
                y0 = int(i * h_pre / n_rows)
                y1 = int((i + 1) * h_pre / n_rows)
                for j in range(n_cols):
                    x0 = int(j * w_pre / n_cols)
                    x1 = int((j + 1) * w_pre / n_cols)
                    patch = pre[y0:y1, x0:x1]

                    val = feature_fn(patch, dataset_key, unpreprocessed=False)
                    # apply rescale, NO clipping
                    val = float(val) * float(rescale)
                    heatmap_sum[i, j] += val

            used += 1

        if used == 0:
            print(f"[{dataset_key}] No readable images for condition {cond_name} ({cond_code}), skipping.")
            continue

        avg_heatmap = heatmap_sum / float(used)
        heatmaps_by_condition[str(cond_code)] = avg_heatmap

        vals = avg_heatmap[np.isfinite(avg_heatmap)]
        if vals.size:
            local_max = float(np.max(vals))
            global_max = max(global_max, local_max)
            print(
                f"[{dataset_key}] {cond_name} | n={used} | "
                f"min={float(np.min(vals)):.6f} mean={float(np.mean(vals)):.6f} "
                f"max={local_max:.6f} p95={float(np.quantile(vals, 0.95)):.6f}"
            )
        else:
            print(f"[{dataset_key}] {cond_name} | n={used} | all values non-finite")

    if not heatmaps_by_condition:
        print(f"[make_condition_average_heatmaps_and_scale] No heatmaps produced for {dataset_key}/{feature_key}.")
        return {}, 0.0, 1.0

    vmin = 0.0
    vmax = max(1.0, float(global_max))

    # Decide tick distance if not provided
    if scale_tick_distance is None:
        # aim for ~5–6 ticks, rounded-ish
        approx = vmax / 5.0
        if approx <= 0:
            scale_tick_distance = 0.1
        else:
            # round to 1-2 sig figs
            pow10 = 10 ** np.floor(np.log10(approx))
            mant = approx / pow10
            if mant < 1.5:
                step = 1.0
            elif mant < 3.5:
                step = 2.0
            elif mant < 7.5:
                step = 5.0
            else:
                step = 10.0
            scale_tick_distance = float(step * pow10)

    # Plot & save each heatmap with shared vmin/vmax
    for cond_code, avg_heatmap in heatmaps_by_condition.items():
        cond_name = condition_map.get(str(cond_code), str(cond_code))

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            avg_heatmap,
            interpolation="nearest",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap_use,
        )
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if show:
            plt.show()
        else:
            plt.close(fig)

        if save_prefix is None:
            base = f"{dataset_key}_{cond_name.replace(' ', '')}_{feature_key}_heatmap"
        else:
            base = f"{save_prefix}_{dataset_key}_{cond_name.replace(' ', '')}_{feature_key}_heatmap"

        base = _safe(base)
        save_path = os.path.join(viz_path, f"{base}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"[heatmap] {dataset_key} | {cond_name} | saved: {save_path}")

    # Create ONE shared scale bar figure
    if scale_save_name is None:
        if save_prefix is None:
            scale_base = f"{dataset_key}_{feature_key}_scale"
        else:
            scale_base = f"{save_prefix}_{dataset_key}_{feature_key}_scale"
    else:
        scale_base = scale_save_name

    scale_base = _safe(scale_base)

    fig, ax = plt.subplots(figsize=(scale_width, scale_height))
    gradient = np.linspace(vmax, vmin, 256).reshape(256, 1)
    ax.imshow(gradient, aspect="auto", cmap=cmap_use, origin="upper")

    ax.set_xticks([])
    ax.set_ylabel(feat_label, fontsize=scale_text_size)

    ticks = np.arange(vmin, vmax + scale_tick_distance, scale_tick_distance, dtype=np.float64)
    ticks = np.clip(ticks, vmin, vmax)

    # map tick values to pixel y (0 at top, 255 at bottom)
    if vmax > vmin:
        y_positions = 255 - ((ticks - vmin) / (vmax - vmin) * 255)
    else:
        y_positions = np.full_like(ticks, 255.0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=scale_text_size)

    for spine in ax.spines.values():
        spine.set_linewidth(scale_border_thickness)

    fig.subplots_adjust(left=0.5, right=0.9, top=0.98, bottom=0.05)

    if show:
        plt.show()
    else:
        plt.close(fig)

    scale_path = os.path.join(viz_path, f"{scale_base}.png")
    fig.savefig(scale_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"[scale] saved: {scale_path} | vmin={vmin:.3f} vmax={vmax:.3f} tick={scale_tick_distance:.3f}")

    return heatmaps_by_condition, vmin, vmax
