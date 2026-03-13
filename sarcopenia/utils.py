import os
import subprocess
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm
from matplotlib.patches import Patch

def save_color_legend(mask_name_to_color, out_file):
    if os.path.exists(out_file):
        return

    legend_elements = [
        Patch(facecolor=color, label=name)
        for name, color in mask_name_to_color.items()
    ]

    fig, ax = plt.subplots(figsize=(4, len(mask_name_to_color)*0.6))
    ax.axis("off")

    ax.legend(
        handles=legend_elements,
        loc="center",
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

def make_gif(png_files, output_gif, duration=0.1, loop=0):
    png_files = sorted(png_files)  # Ensure files are in the correct order
    images = [imageio.v2.imread(f) for f in png_files]
    imageio.mimsave(output_gif, images, duration=duration, loop=loop)

def visualize_segmentations(
    image_file,
    segmentation_files,
    output_dir,
    case_id="case",
    overwrite=False,
    duration=0.1,
    loop=0,
):
    """Create per-slice overlays for a set of segmentation masks."""
    if not segmentation_files:
        print(f"No segmentation files to visualize for {case_id}.")
        return

    image_file = Path(image_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_nii = nib.load(str(image_file))
    img = img_nii.get_fdata()
    img = np.rot90(img, axes=(0, 1))

    if isinstance(segmentation_files, (str, Path)):
        segmentation_files = [segmentation_files]

    masks = []
    for seg_path in segmentation_files:
        seg_path = Path(seg_path)
        if not seg_path.exists():
            continue
        mask_nii = nib.load(str(seg_path))
        mask = mask_nii.get_fdata()
        mask = np.rot90(mask, axes=(0, 1))
        masks.append((seg_path.stem.replace(".nii", ""), mask > 0))

    if not masks:
        print(f"All segmentation files were missing for {case_id}.")
        return

    # Aggregate all structures into one visualization mask.
    colors = np.array([
        [255, 0, 0],      # red
        [0, 255, 0],      # green
        [0, 0, 255],      # blue
        [255, 255, 0],    # yellow
        [255, 0, 255],    # magenta
        [0, 255, 255],    # cyan
    ]) / 255.0
    
    mask_name_to_color = {}
    combined_mask = np.zeros_like(masks[0][1], dtype=int)
    for i, (mask_name, mask) in enumerate(masks, start=1):
        combined_mask[mask > 0] = i  # Assign a unique integer to each structure for coloring    
        mask_name_to_color[mask_name] = colors[(i-1) % len(colors)]

    color_legend_file = output_dir / f"{case_id}_legend.png"
    save_color_legend(mask_name_to_color, color_legend_file)

    png_files = []
    for z in tqdm(range(img.shape[2]), desc=f"Visualizing {case_id}"):
        base = img[:, :, z]
        slice_mask = combined_mask[:, :, z]

        n_mask_pixels = int(np.sum(slice_mask > 0))

        out_path = output_dir / f"{case_id}_slice{z:03d}.png"
        if n_mask_pixels > 0:
            out_path = output_dir / f"{case_id}_slice{z:03d}_K.png"

        png_files.append(out_path)

        if out_path.exists() and not overwrite:
            continue

        # normalize image for RGB
        base_norm = (base - base.min()) / (base.max() - base.min() + 1e-8)
        rgb = np.stack([base_norm]*3, axis=-1)

        alpha = 0.35

        # overlay masks
        for label in np.unique(slice_mask):
            if label == 0:
                continue

            mask_region = slice_mask == label
            color = colors[(label-1) % len(colors)]

            rgb[mask_region] = (
                (1-alpha)*rgb[mask_region] +
                alpha*color
            )

        fig, axes = plt.subplots(1,2,figsize=(12,6))

        axes[0].imshow(base, cmap="gray")
        axes[0].axis("off")

        axes[1].imshow(rgb)
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
    
    gif_file = output_dir / f"{case_id}_visualization.gif"
    if not gif_file.exists() or overwrite:
        make_gif(png_files, gif_file, duration=duration, loop=loop)