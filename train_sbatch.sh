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

usage() {
    cat <<'EOF'
Usage:
  train_sbatch.sh [baseline]
  train_sbatch.sh compare --final-test DIR --test-csv FILE --selection-out FILE [--matched-results DIR]
  train_sbatch.sh ablation --selection FILE --cross-dataset DIR --test-csv FILE
  train_sbatch.sh compare-and-ablate --final-test DIR --test-csv FILE --selection-out FILE --cross-dataset DIR [--matched-results DIR]
  train_sbatch.sh baseline-then-compare [compare options]
  train_sbatch.sh baseline-then-compare-and-ablate [compare-and-ablate options]

The comparison and ablation modes evaluate existing predictions with Python 3.12.
Set REPORT_PYTHON to select the Python executable for these modes.
EOF
}

mode=${1:-baseline}
if (( $# > 0 )); then
    shift
fi

run_report() (
        set -euo pipefail
        mode=$1
        shift
        cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
        report_python=${REPORT_PYTHON:-python3}
        "$report_python" -c 'import sys; sys.exit("Report evaluation requires Python 3.12 or later.") if sys.version_info < (3, 12) else None'
        case "$mode" in
            compare) exec "$report_python" src/evaluate_transformer_results.py "$@" ;;
            ablation) exec "$report_python" src/evaluate_transformer_ablation.py "$@" ;;
        esac

        comparison_args=()
        cross_dataset=
        selection_out=
        test_csv=
        while (( $# )); do
            case "$1" in
                --final-test|--test-csv|--selection-out|--cross-dataset|--matched-results)
                    if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
                        echo "Missing value for $1" >&2
                        exit 2
                    fi
                    case "$1" in
                        --cross-dataset) cross_dataset=$2 ;;
                        --selection-out) selection_out=$2 ;;
                        --test-csv) test_csv=$2 ;;
                    esac
                    if [[ "$1" != --cross-dataset ]]; then
                        comparison_args+=("$1" "$2")
                    fi
                    shift 2
                    ;;
                *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
            esac
        done
        if [[ -z "$cross_dataset" || -z "$test_csv" || -z "$selection_out" ]]; then
            echo "compare-and-ablate requires --cross-dataset, --test-csv and --selection-out" >&2
            exit 2
        fi
        "$report_python" src/evaluate_transformer_results.py "${comparison_args[@]}"
        exec "$report_python" src/evaluate_transformer_ablation.py \
            --selection "$selection_out" --cross-dataset "$cross_dataset" --test-csv "$test_csv"
)

case "$mode" in
    -h|--help) usage; exit 0 ;;
    compare|ablation|compare-and-ablate) run_report "$mode" "$@"; exit $? ;;
    baseline-then-compare|baseline-then-compare-and-ablate)
        if (( $# == 0 )); then
            usage >&2
            exit 2
        fi
        ;;
    baseline)
        if (( $# )); then
            usage >&2
            exit 2
        fi
        ;;
    *) usage >&2; exit 2 ;;
esac

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
  --max_seq_length 128 || exit $?

if [[ "$mode" == baseline-then-* ]]; then
    run_report "${mode#baseline-then-}" "$@" || exit $?
fi

echo "=========================================================="
echo "Job Completed at: $(date)"
echo "=========================================================="
