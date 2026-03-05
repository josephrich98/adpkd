import os
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage import morphology
from scipy.ndimage import distance_transform_edt
from utils import visualize_segmentations


CRISP_ROOT = Path("/mnt/gpussd2/jrich/Desktop/ADPKD/crisp/T2_HASTE")
SEGMENTATION_OUTPUT_DIR = Path("/mnt/gpussd2/jrich/Desktop/ADPKD/segmentations")
SEGMENTATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_intensity(img):
    """Robust MRI intensity normalization."""
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    img = (img - p1) / (p99 - p1 + 1e-8)
    img = np.clip(img, 0, 1)
    return img


def segment_slice(slice_img):
    """
    Segment SAT, VAT, and muscle from a single 2D slice.

    Returns label mask:
        0 = background
        1 = SAT
        2 = VAT
        3 = muscle
    """

    # --- body mask ---
    body = slice_img > 0.05
    body = morphology.remove_small_objects(body, 500)
    body = morphology.binary_closing(body, morphology.disk(5))

    # --- fat detection ---
    fat = slice_img > 0.6
    fat = fat & body

    # --- distance from body boundary ---
    dist = distance_transform_edt(body)

    sat = fat & (dist < 20)
    vat = fat & (dist >= 20)

    # --- muscle estimate ---
    muscle = body & (~fat) & (slice_img > 0.2) & (slice_img < 0.6)
    muscle = morphology.remove_small_objects(muscle, 200)

    labels = np.zeros(slice_img.shape, dtype=np.uint8)
    labels[sat] = 1
    labels[vat] = 2
    labels[muscle] = 3

    return labels


def process_case(image_file, output_file):
    nii = nib.load(image_file)
    img = nii.get_fdata()

    img = normalize_intensity(img)

    seg = np.zeros(img.shape, dtype=np.uint8)

    for z in range(img.shape[2]):
        seg[:, :, z] = segment_slice(img[:, :, z])

    seg_nii = nib.Nifti1Image(seg, nii.affine, nii.header)
    nib.save(seg_nii, output_file)


for patientID in tqdm(sorted(os.listdir(CRISP_ROOT)), desc="Processing patients"):
    patient_dir = CRISP_ROOT / patientID
    if not patient_dir.is_dir():
        continue

    for caseID in sorted(os.listdir(patient_dir)):
        case_dir = patient_dir / caseID

        image_file = None
        for filename in os.listdir(case_dir):
            if filename.endswith(".nii.gz"):
                image_file = case_dir / filename
                break

        if image_file is None:
            print(f"No NIfTI file found for caseID {caseID} in {case_dir}. Skipping.")
            continue

        output_case_dir = SEGMENTATION_OUTPUT_DIR / "custom" / patientID / caseID
        output_case_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_case_dir / f"{caseID}_body_composition_seg.nii.gz"

        if output_file.exists():
            continue

        try:
            process_case(image_file, output_file)
        except Exception as e:
            print(f"Failed for {image_file}: {e}")
        
        for plane in ["coronal", "axial"]:
            visualize_segmentations(
                image_file=image_file,
                segmentation_files=output_file,
                output_dir=output_case_dir / "visualization" / plane,
                case_id=caseID,
                duration=0.1,
                loop=0,
                plane=plane,
            )