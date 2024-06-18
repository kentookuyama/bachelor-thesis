#!/usr/bin/env bash
#
#SBATCH --job-name FullStrategies
#SBATCH --output=all_strategies_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --mincpus=10

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

# Define the directories and paths
declare -A dirs=(
  ["./xview2"]="/home/stud/okuyama/data/xview2"
  ["./LEVIR-CD"]="/home/stud/okuyama/data/LEVIR-CD"
)

# Check and create symbolic links
for path in "${!dirs[@]}"; do
  dest="${dirs[$path]}"
  if [ ! -e "$path" ]; then
    echo "Creating symbolic link: $path -> $dest"
    ln -s "$dest" "$path"
  else
    echo "Symbolic link already exists: $path -> $dest"
  fi
done

# Run your specific script
config_path='trainxView2.r50_farseg_changemixin_full'
model_dir='./log/changestar_sisup/r50_farseg_changemixin_full'

export LOCAL_RANK=0

torchrun --nproc_per_node=${NUM_GPUS} --master_port=29502 ./train_changemixin.py \
  --config_path=${config_path} \
  --model_dir=${model_dir}