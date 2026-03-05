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

