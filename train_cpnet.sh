#!/bin/bash

# Optional: stop if error occurs
set -e


# Activate virtual environment if needed
module load conda
conda activate rlcv

# Run the script
torchrun --nproc_per_node=8 train.py --lr 1e-4 --cv_dir checkpoints/policy_grad/ --img_size 172 --batch_size 64 --data_dir data/caltech/ --alpha 0.8 --beta 0.005 --sigma 0.02 --max_epochs 1450