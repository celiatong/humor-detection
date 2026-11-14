# HKUST Superpod Deep Learning: Humor Detection 🚀

This repository contains the fine-tuning pipeline for humor detection in Product Question Answering (PQA) datasets using **DeBERTa-v3** natively in Keras 3 / TensorFlow.

This setup is fully optimized for **NVIDIA H800 (80GB) GPUs** on the HKUST Superpod, implementing dynamic memory allocation, Tensor-Core-powered **Mixed Precision (float16)**, **Cosine Decay learning rate schedules**, and **Binary Focal Crossentropy Loss**.

---

## SuperPOD setup

```bash
git clone git@github.com:celiatong/humor-detection.git
cd humor-detection

conda env create -f environment.yml

conda activate humor
mkdir -p logs results
```

## Run a job

```bash
sbatch train_sbatch.sh        # Original DeBERTa training
bash train_sbatch.sh --help  # Report modes and required arguments
```

- `compare` ranks the three-seed results, saves the winner, and stops.
- `ablation` reads the saved winner and rebuilds its ablation table.
- `compare-and-ablate` runs both stages. Prefix `compare` or
  `compare-and-ablate` with `baseline-then-` to train first.

Report modes use Python 3.12+ and saved predictions. Supply the original
`test.csv`, the prediction archives, and optionally `--matched-results results`
for the local three-seed runs. Set `REPORT_PYTHON` if `python3` is older.

For PyTorch training options, run `python src/train_transformer_pytorch.py --help`.

## Monitor a job

```bash
squeue -u "$USER"
tail -f logs/job_<JOB_ID>.out
scancel <JOB_ID>
```
