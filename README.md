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