#!/bin/bash

#SBATCH --job-name=deberta_humor
#SBATCH --account=mscbdt2024
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

echo "=========================================================="
echo "Job Execution ID:    $SLURM_JOB_ID"
echo "Allocated Node:      $(hostname)"
echo "Submission Dir:      $SLURM_SUBMIT_DIR"
echo "Start Time:          $(date)"
echo "=========================================================="

# Display GPU Allocation State
nvidia-smi

# Load system CUDA if available (as fallback)
module load cuda 2>/dev/null || module load cuda/12.2 2>/dev/null || true

# Activate Conda Environment
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook 2>/dev/null)"
fi

conda activate humor

# ==============================================================================
# 🚀 FIX: Dynamic path discovery of Conda-installed CUDA/cuDNN libraries
# ==============================================================================
echo "[INFO] Configuring LD_LIBRARY_PATH to avoid dlopen warnings..."

# 1. Fetch site-packages location for the active environment
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# 2. Gather library endpoints for deep-learning runtime binaries (cuDNN, cuBLAS, cuFFT, etc.)
NVIDIA_LIBS=$(find $SITE_PACKAGES/nvidia/*/lib/ -maxdepth 0 -type d 2>/dev/null | paste -sd ":" -)

# 3. Export to LD_LIBRARY_PATH dynamically
if [ ! -z "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH=$NVIDIA_LIBS:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    echo "[INFO] Dynamic library resolution successful."
else
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    echo "[WARN] Custom libraries missing. Reverted to standard Conda prefix."
fi

# Ensure Python includes local src folder
export PYTHONPATH="${PYTHONPATH}:./src"

# ==============================================================================

# Verify TensorFlow GPU Detection state
echo "[INFO] Running GPU Detection Pre-Flight Check..."
echo "[INFO] Active Python: $(which python)"
python -c "import tensorflow as tf; print('[INFO] GPUs Detected by TensorFlow:', tf.config.list_physical_devices('GPU'))"

# Prepare Directories
mkdir -p data saved_models results logs

# Run Training Pipeline
python src/train_transformer.py \
  --data_dir ./data \
  --output_dir ./saved_models \
  --results_dir ./results \
  --batch_size 128 \
  --lr 2e-5 \
  --epochs 10 \
  --max_seq_length 128

echo "=========================================================="
echo "Job Completed at: $(date)"
echo "=========================================================="
