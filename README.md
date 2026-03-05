Ensure I have model_checkpoint, batch_inference.py, inference.py, requirements.txt in the same folder - see https://drive.google.com/drive/folders/1SiAMIj70cX0gbYOwYsSjlpcMtfq6bvWT

conda create -n adpkd python=3.10
conda activate adpkd
pip install -r requirements.txt
pip3 install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu126
- might need to update based on CUDA version - https://pytorch.org/get-started/locally/
(assume I'm in my project directory)
export nnUNet_raw_data_base=$PWD
export nnUNet_preprocessed=$PWD
export RESULTS_FOLDER=$PWD

ensure data is in crisp/T2_HASTE
update paths in batch_inference.py

output will be in crisp/T2_HASTE_segmentations


cd ..
conda create -n medsam python=3.10 -y
conda activate medsam
install pytorch:
- Mac: pip3 install torch torchvision
- radgpu1: pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/bowang-lab/MedSAM
cd MedSAM
pip install -e .
pip install PyQt5 gdown
cd work_dir/MedSAM
<!-- wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -->
gdown --id 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O medsam_vit_b.pth
python gui.py