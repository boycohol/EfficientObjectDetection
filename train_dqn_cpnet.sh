#!/bin/bash

# Optional: stop if error occurs
set -e


# Activate virtual environment if needed
module load conda
conda activate rlcv

# Run the script
torchrun --nproc_per_node=8 train_dqn.py --lr 1e-4 --cv_dir checkpoints/dqn/ --img_size 172 --batch_size 64 --data_dir data/caltech/ --beta 0.005 --sigma 0.02 --max_epochs 1450