import shutil
import subprocess
from pathlib import Path

CRISP_ROOT = Path("/mnt/gpussd2/jrich/Desktop/ADPKD/crisp/T2_HASTE") # Get data from here
SEG_ROOT = Path("/mnt/gpussd2/jrich/Desktop/ADPKD/crisp/T2_HASTE_segmentations") # CHANGE THIS: output directory

PROJECT_ROOT = Path("/mnt/gpussd2/jrich/Desktop/ADPKD") # CHANGE THIS: project root
MODEL_CHECKPOINT = PROJECT_ROOT / "model_checkpoint"
INFERENCE_SCRIPT = PROJECT_ROOT / "inference.py"

TMP_ROOT = Path("/mnt/gpussd2/jrich/Desktop/ADPKD/tmp_batch_inference") # CHANGE THIS: temporary directory
TMP_INPUT = TMP_ROOT / "images"
TMP_OUTPUT = TMP_ROOT / "segmentations"


def clear_tmp():
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    for p in TMP_ROOT.glob("*"):
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def run_inference_on_nifti(nifti_path: Path):
    print(f"\nProcessing: {nifti_path}")

    clear_tmp()
    TMP_INPUT.mkdir(parents=True, exist_ok=True)
    TMP_OUTPUT.mkdir(parents=True, exist_ok=True)

    tmp_nifti = TMP_INPUT / nifti_path.name
    shutil.copy2(nifti_path, tmp_nifti)

    cmd = [
        "python",
        str(INFERENCE_SCRIPT),
        "--model_checkpoint", str(MODEL_CHECKPOINT),
        "--input_dir", str(TMP_INPUT),
        "--output_dir", str(TMP_OUTPUT),
    ]

    subprocess.run(cmd, check=True)

    seg_files = list(TMP_OUTPUT.glob("*.nii.gz"))
    if len(seg_files) != 1:
        raise RuntimeError(f"Expected 1 segmentation, found {len(seg_files)}")

    seg_path = seg_files[0]

    relative = nifti_path.relative_to(CRISP_ROOT)
    final_seg_path = SEG_ROOT / relative.parent / f"{nifti_path.stem}_seg.nii.gz"
    final_seg_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(seg_path, final_seg_path)

    print(f"Saved segmentation → {final_seg_path}")


def main():
    nifti_files = list(CRISP_ROOT.rglob("*.nii.gz"))
    print(f"Found {len(nifti_files)} NIfTI files")

    for nifti in nifti_files:
        try:
            run_inference_on_nifti(nifti)
        except Exception as e:
            print(f"FAILED on {nifti}: {e}")


if __name__ == "__main__":
    main()
