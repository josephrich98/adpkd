import os
import subprocess
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

def make_gif(png_files, output_gif, duration=0.1, loop=0):
    png_files = sorted(png_files)  # Ensure files are in the correct order
    images = [imageio.v2.imread(f) for f in png_files]
    imageio.mimsave(output_gif, images, duration=duration, loop=loop)

def visualize_segmentations(
    image_file,
    segmentation_files,
    output_dir,
    case_id,
    overwrite=False,
    duration=0.1,
    loop=0,
    plane="coronal",
):
    """Create per-slice overlays for a set of segmentation masks."""
    if not segmentation_files:
        print(f"No segmentation files to visualize for {case_id}.")
        return

    image_file = Path(image_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_nii = nib.load(str(image_file))
    if plane == "coronal":
        pass
    elif plane == "axial":
        img_nii = nib.as_closest_canonical(img_nii)
    else:
        raise ValueError(f"Unsupported plane: {plane}")
    img = img_nii.get_fdata()
    img = np.rot90(img, axes=(0, 1))

    masks = []
    for seg_path in segmentation_files:
        seg_path = Path(seg_path)
        if not seg_path.exists():
            continue
        mask_nii = nib.load(str(seg_path))
        if plane == "coronal":
            pass
        elif plane == "axial":
            mask_nii = nib.as_closest_canonical(mask_nii)
        else:
            raise ValueError(f"Unsupported plane: {plane}")
        mask = mask_nii.get_fdata()
        mask = np.rot90(mask, axes=(0, 1))
        masks.append((seg_path.stem.replace(".nii", ""), mask > 0))

    if not masks:
        print(f"All segmentation files were missing for {case_id}.")
        return

    # Aggregate all structures into one visualization mask.
    combined_mask = np.zeros_like(masks[0][1], dtype=bool)
    for _, mask in masks:
        combined_mask |= mask
    
    # # ensure same orientation and shape between img and combined_mask
    # if img.shape != combined_mask.shape:
    #     print(f"Shape mismatch between image and combined mask for {case_id}. Skipping visualization.")
    #     return
    
    # img_orientation = nib.orientations.aff2axcodes(img_nii.affine)
    # mask_orientation = nib.orientations.aff2axcodes(mask_nii.affine)
    # print(f"Image orientation for {case_id}: {img_orientation}")
    # print(f"Mask orientation for {case_id}: {mask_orientation}")
    # if img_orientation != mask_orientation:
    #     print(f"Orientation mismatch between image and mask for {case_id}. Skipping visualization.")
    #     return

    png_files = []
    for z in tqdm(range(img.shape[2]), desc=f"Visualizing {case_id}"):
        n_mask_pixels = int(np.sum(combined_mask[:, :, z]))

        out_path = output_dir / f"{case_id}_slice{z:03d}.png"
        if n_mask_pixels > 0:
            out_path = output_dir / f"{case_id}_slice{z:03d}_K.png"
        
        png_files.append(out_path)

        if out_path.exists() and not overwrite:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(img[:, :, z], cmap="gray")
        axes[0].set_title("Image only")
        axes[0].axis("off")

        axes[1].imshow(img[:, :, z], cmap="gray")
        axes[1].imshow(combined_mask[:, :, z], cmap="Reds", alpha=0.25)
        axes[1].set_title(f"Image + mask ({n_mask_pixels} pixels)")
        axes[1].axis("off")

        plt.suptitle(f"{case_id} | {', '.join(name for name, _ in masks)} | Slice {z}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
    
    gif_file = output_dir / f"{case_id}_visualization.gif"
    if not gif_file.exists() or overwrite:
        make_gif(png_files, gif_file, duration=duration, loop=loop)