# HKUST Superpod Deep Learning: Humor Detection 🚀

This repository contains the fine-tuning pipeline for humor detection in Product Question Answering (PQA) datasets using **DeBERTa-v3** natively in Keras 3 / TensorFlow.

This setup is fully optimized for **NVIDIA H800 (80GB) GPUs** on the HKUST Superpod, implementing dynamic memory allocation, Tensor-Core-powered **Mixed Precision (float16)**, **Cosine Decay learning rate schedules**, and **Binary Focal Crossentropy Loss**.

---

## 🛠️ Step 1: Rebuilding the Conda Environment

To perfectly replicate the environment, package versions, and underlying CUDA libraries, rebuild the environment using Conda:

```bash
# 1. Clone the repository and navigate inside
git clone git@github.com:celiatong/humor-detection.git
cd humor-detection

# 2. Build the environment from the universal blueprint
conda env create -f environment.yml

# 3. Activate your new environment
conda activate humor

## 🛠️ Step 2: Summit job:

sbatch train_sbatch.sh


## 🛠️ Step 3: Handy Slurm Commands:
Check your job's queue status:

 bash squeue -u $USER

Terminate/Stop a running job:

 bash scancel <JOB_ID>

Inspect real-time execution outputs:

 bash tail -f logs/job_<JOB_ID>.out
