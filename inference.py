import numpy as np
from torch.serialization import add_safe_globals

add_safe_globals([
    np.dtype,
    np.core.multiarray.scalar,
])

import os
import glob
import argparse
import shutil
import contextlib
from tqdm import tqdm

from nnunet.inference.predict import predict_cases

def parse_args():
    p = argparse.ArgumentParser(description="ADPKD nnU-Net inference")
    p.add_argument(
        "--model_checkpoint",
        required=True,
        help="Path to nnU-Net model checkpoint directory"
    )
    p.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing input .nii.gz files"
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write segmentation outputs"
    )
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    nii_file_list = glob.glob(os.path.join(args.input_dir, "*.nii.gz"))

    if len(nii_file_list) == 0:
        raise RuntimeError(
            "No NIfTI files found in input_dir. "
            "Colab test-data fallback removed for HPC safety."
        )

    output_path_list = [
        os.path.join(args.output_dir, os.path.basename(i))
        for i in nii_file_list
    ]

    input_output_pair = dict(zip(nii_file_list, output_path_list))

    for input_path, output_path in tqdm(input_output_pair.items()):
        with contextlib.redirect_stdout(None):
            predict_cases(
                args.model_checkpoint,
                [[input_path]],
                [output_path],
                num_threads_preprocessing=6,
                num_threads_nifti_save=2,
                do_tta=True,
                mixed_precision=True,
                overwrite_existing=True,
                all_in_gpu=None,
                step_size=0.5,
                checkpoint_name="model_final_checkpoint",
                segmentation_export_kwargs=None,
                disable_postprocessing=True,
                folds="all",
                save_npz=False
            )


if __name__ == "__main__":
    main()
