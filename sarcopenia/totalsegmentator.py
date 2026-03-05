import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

# psoas
# paraspinal muscles
# abdominal wall muscles
# visceral fat
# subcutaneous fat

CRISP_ROOT = Path("/mnt/gpussd2/jrich/Desktop/ADPKD/crisp/T2_HASTE")  # Get data from here
SEGMENTATION_OUTPUT_DIR = Path("/mnt/gpussd2/jrich/Desktop/ADPKD/segmentations")  # Save segmentations here

selected_segmentations_total_mr = ["iliopsoas_left", "iliopsoas_right"]
selected_segmentations_tissue_types_mr = ["subcutaneous_fat", "torso_fat", "skeletal_muscle"]

def visualize_segmentations(
    image_file,
    segmentation_files,
    output_dir,
    case_id,
    overwrite=False,
):
    """Create per-slice overlays for a set of segmentation masks."""
    if not segmentation_files:
        print(f"No segmentation files to visualize for {case_id}.")
        return

    image_file = Path(image_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(image_file)).get_fdata()
    img = np.rot90(img, axes=(0, 1))

    masks = []
    for seg_path in segmentation_files:
        seg_path = Path(seg_path)
        if not seg_path.exists():
            continue
        mask = nib.load(str(seg_path)).get_fdata()
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
    
    # img_orientation = nib.orientations.aff2axcodes(nib.load(str(image_file)).affine)
    # mask_orientation = nib.orientations.aff2axcodes(nib.load(str(segmentation_files[0])).affine)
    # print(f"Image orientation for {case_id}: {img_orientation}")
    # print(f"Mask orientation for {case_id}: {mask_orientation}")
    # if img_orientation != mask_orientation:
    #     print(f"Orientation mismatch between image and mask for {case_id}. Skipping visualization.")
    #     return

    for z in tqdm(range(img.shape[2]), desc=f"Visualizing {case_id}"):
        n_mask_pixels = int(np.sum(combined_mask[:, :, z]))

        out_path = output_dir / f"{case_id}_slice{z:03d}.png"
        if n_mask_pixels > 0:
            out_path = output_dir / f"{case_id}_slice{z:03d}_K.png"

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

        plt.suptitle(f"{case_id} | {', '.join(name for name, _ in masks)}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close(fig)


def existing_segmentation_paths(seg_dir, segmentation_names):
    return [
        Path(seg_dir) / f"{name}.nii.gz"
        for name in segmentation_names
        if (Path(seg_dir) / f"{name}.nii.gz").exists()
    ]

def run_totalsegmentator(task, image_file, selected_segmentations):
    totalsegmentator_dir = SEGMENTATION_OUTPUT_DIR / "totalsegmentator" / patientID / caseID / task
    totalsegmentator_dir.mkdir(parents=True, exist_ok=True)

    totalsegmentator_command = [
        "TotalSegmentator",
        "-i",
        image_file,
        "-o",
        str(totalsegmentator_dir),
        "--task",
        task,
    ]
    if all((totalsegmentator_dir / f"{seg_name}.nii.gz").exists() for seg_name in selected_segmentations):
        print(f"Predicted {task} segmentation files already exist for caseID {caseID}. Skipping TotalSegmentator.")
    else:
        print(f"Running TotalSegmentator {task} for caseID {caseID}...")
        subprocess.run(totalsegmentator_command, check=True)

    visualize_segmentations(
        image_file=image_file,
        segmentation_files=existing_segmentation_paths(totalsegmentator_dir, selected_segmentations),
        output_dir=totalsegmentator_dir / f"visualization_{task}",
        case_id=caseID,
    )

for patientID in tqdm(sorted(os.listdir(CRISP_ROOT)), desc="Processing patients"):
    patient_dir = os.path.join(CRISP_ROOT, patientID)
    if not os.path.isdir(patient_dir):
        continue
    for caseID in sorted(os.listdir(patient_dir)):
        # get the file that ends in .nii.gz
        image_file = None
        case_dir = os.path.join(CRISP_ROOT, patientID, caseID)
        for filename in os.listdir(case_dir):
            if filename.endswith(".nii.gz"):
                image_file = os.path.join(case_dir, filename)
                break
        if image_file is None:
            print(f"No NIfTI file found for caseID {caseID} in {case_dir}. Skipping.")
            continue

        # * run TotalSegmentator
        run_totalsegmentator(task="total_mr", image_file=image_file, selected_segmentations=selected_segmentations_total_mr)
        run_totalsegmentator(task="tissue_types_mr", image_file=image_file, selected_segmentations=selected_segmentations_tissue_types_mr)