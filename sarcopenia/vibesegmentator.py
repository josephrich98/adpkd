import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from utils import visualize_segmentations

# sacrum
# muscle
# inner fat
# subcutaneous fat

CRISP_ROOT = Path("/mnt/gpussd2/jrich/Desktop/ADPKD/crisp/T2_HASTE")  # Get data from here
SEGMENTATION_OUTPUT_DIR = Path("/mnt/gpussd2/jrich/Desktop/ADPKD/segmentations")  # Save segmentations here
VIBESegmentator = "/mnt/gpussd2/jrich/Desktop/VIBESegmentator"  # Path to VIBESegmentator repository

selected_segmentations = {
    "sacrum": 23,
    "subcutaneous_fat": 65,
    "muscle": 66,
    "inner_fat": 67,

}

def existing_segmentation_paths(seg_dir, segmentation_names):
    return [
        Path(seg_dir) / f"{name}.nii.gz"
        for name in segmentation_names
        if (Path(seg_dir) / f"{name}.nii.gz").exists()
    ]

def run_vibesegmentator(image_file):
    vibesegmentator_dir = SEGMENTATION_OUTPUT_DIR / "vibesegmentator" / patientID / caseID
    vibesegmentator_dir.mkdir(parents=True, exist_ok=True)

    vibesegmentator_out_nii_filename = os.path.basename(image_file).replace(".nii.gz", "_vibesegmentator_output.nii.gz")
    vibesegmentator_out_nii = vibesegmentator_dir / "vibesegmentator_output.nii.gz"

    VIBESegmentator_script = os.path.join(VIBESegmentator, "run_VIBESegmentator.py")
    vibesegmentator_command = [
        "python",
        VIBESegmentator_script,
        "--img",
        image_file,
        "--out_path",
        str(vibesegmentator_out_nii),
        "--ddevice",
        "cuda",
        "--dataset_id",
        "100",
        "--fill_holes"
    ]

    if os.path.exists(vibesegmentator_out_nii_filename):
        print(f"Predicted segmentation file already exists for caseID {caseID}. Skipping VIBESegmentator.")
    else:
        print(f"Running VIBESegmentator for caseID {caseID}...")
        print(" ".join(vibesegmentator_command))
        subprocess.run(vibesegmentator_command, check=True)

for patientID in tqdm(sorted(os.listdir(CRISP_ROOT)), desc="Processing patients"):
    patient_dir = os.path.join(CRISP_ROOT, patientID)
    if not os.path.isdir(patient_dir):
        continue
    for caseID in sorted(os.listdir(patient_dir)):
        case_dir = os.path.join(CRISP_ROOT, patientID, caseID)
        if not os.path.isdir(case_dir):
            continue

        # get the file that ends in .nii.gz
        image_file = None
        for filename in os.listdir(case_dir):
            if filename.endswith(".nii.gz"):
                image_file = os.path.join(case_dir, filename)
                break
        if image_file is None:
            print(f"No NIfTI file found for caseID {caseID} in {case_dir}. Skipping.")
            continue

        # * run VIBESegmentator
        run_vibesegmentator(image_file=image_file)

        segmentation_files = None
        vibesegmentator_base_dir = SEGMENTATION_OUTPUT_DIR / "vibesegmentator" / patientID / caseID
        
        vibesegmentator_dir = SEGMENTATION_OUTPUT_DIR / "vibesegmentator" / patientID / caseID
        vibesegmentator_out_nii_filename = os.path.basename(image_file).replace(".nii.gz", "_vibesegmentator_output.nii.gz")
        vibesegmentator_out_nii = vibesegmentator_dir / "vibesegmentator_output.nii.gz"

        visualize_segmentations(
            image_file=image_file,
            segmentation_files=vibesegmentator_out_nii,
            output_dir=vibesegmentator_base_dir / "visualization",
            case_id=f"{patientID}___{caseID}",
            duration=0.1,
            loop=0,
            label_dict=selected_segmentations,
        )

        break
    break