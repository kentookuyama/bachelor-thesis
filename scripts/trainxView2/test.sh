#!/usr/bin/env bash
#
#SBATCH --job-name ppc
#SBATCH --output=ppc.txt
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mincpus=8

# debug info
hostname
which python3
nvidia-smi

python3 -V
env

# Activate Miniconda environment
source /home/stud/okuyama/miniconda3/etc/profile.d/conda.sh

# Activate existing Conda environment
conda activate test_env

pip install -U pip setuptools wheel

# Test CUDA
echo "----------------------------------------"
python -c "import torch; print(f'| Currently CUDA availability: {torch.cuda.is_available():<3} |')"
python -c "import torch; print(f'| Number of CUDA devices: {torch.cuda.device_count():<3} |')"
python -c "import torch; print(f'| Current CUDA device: {torch.cuda.current_device():<3} |')"
echo "----------------------------------------"

# Set environment variables for your script
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

export LOCAL_RANK=0

python3 /home/stud/okuyama/original-changestar/ChangeStar/pcc.py