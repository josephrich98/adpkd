import os
from tqdm import tqdm

crisp_base = "/mnt/gpussd2/jrich/Desktop/ADPKD/crisp"  # {crisp_base}/T2_HASTE/<patient_id>/<file>.nii_seg.nii.gz.seg.nrrd, <file>.nii.gz
medsam_repository_path = "/mnt/gpussd2/jrich/Desktop/MedSAM"
medsam_env = "medsam"

# loop through every subdir in crisp_base/T2_HASTE

for patient_id in tqdm(os.listdir(os.path.join(crisp_base, "T2_HASTE")), desc="Processing patients"):
    patient_dir = os.path.join(crisp_base, "T2_HASTE", patient_id)
    if not os.path.isdir(patient_dir):
        continue
    
    # loop through every file in patient_dir
    for file in os.listdir(patient_dir):
        if file.endswith(".nii.gz"):
            nii_path = os.path.join(patient_dir, file)
            seg_path = nii_path.replace(".nii.gz", ".nii_seg.nii.gz.seg.nrrd")

            # check if seg_path exists
            if not os.path.exists(seg_path):
                print(f"Missing segmentation for {nii_path}, skipping.")
                continue
            
            # run MedSAM inference
            command = f"conda run -n {medsam_env} python {os.path.join(medsam_repository_path, 'MedSAM_Inference.py')} --data_path {nii_path} --seg_path {patient_dir} --box '[95, 255, 190, 350]' --checkpoint {os.path.join(medsam_repository_path, 'work_dir/MedSAM/medsam_vit_b.pth')}"
            os.system(command)