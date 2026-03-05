import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from utils import visualize_segmentations

# psoas
# paraspinal muscles
# abdominal wall muscles
# visceral fat
# subcutaneous fat

CRISP_ROOT = Path("/mnt/gpussd2/jrich/Desktop/ADPKD/crisp/T2_HASTE")  # Get data from here
SEGMENTATION_OUTPUT_DIR = Path("/mnt/gpussd2/jrich/Desktop/ADPKD/segmentations")  # Save segmentations here

selected_segmentations_total_mr = ["iliopsoas_left", "iliopsoas_right"]
selected_segmentations_tissue_types_mr = ["subcutaneous_fat", "torso_fat", "skeletal_muscle"]

def existing_segmentation_paths(seg_dir, segmentation_names):
    return [
        Path(seg_dir) / f"{name}.nii.gz"
        for name in segmentation_names
        if (Path(seg_dir) / f"{name}.nii.gz").exists()
    ]

def run_totalsegmentator(task, image_file, selected_segmentations, duration=0.1, loop=0):
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

    for plane in ["coronal", "axial"]:
        visualize_segmentations(
            image_file=image_file,
            segmentation_files=existing_segmentation_paths(totalsegmentator_dir, selected_segmentations),
            output_dir=totalsegmentator_dir / f"visualization_{task}" / plane,
            case_id=caseID,
            duration=duration,
            loop=loop,
            plane=plane,
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
        run_totalsegmentator(task="total_mr", image_file=image_file, selected_segmentations=selected_segmentations_total_mr, duration=0.1, loop=0)
        run_totalsegmentator(task="tissue_types_mr", image_file=image_file, selected_segmentations=selected_segmentations_tissue_types_mr, duration=0.1, loop=0)