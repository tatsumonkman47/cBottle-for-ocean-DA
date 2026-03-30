#!/bin/bash
#
#SBATCH --job-name="ocean-diff-train"
#SBATCH --output=/home/tm3076/shaferlab/tatsu/cBottle-for-ocean-DA/ocean-examples/train_ocean_diffusion.log
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=60GB
#SBATCH --time=1-00:00:00
#SBATCH --account=torch_pr_432_courant

export WANDB_SILENT=true

singularity exec --nv \
--bind /opt/slurm:/opt/slurm \
--bind /var/run/munge:/var/run/munge \
--overlay /scratch/$USER/singularity_container/overlay-50G-10M.ext3:ro \
/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
/bin/bash -c 'export PATH="/opt/slurm/bin:$PATH" && unset XLA_FLAGS && unset CUDA_CACHE_PATH && source /ext3/env.sh && cd /home/tm3076/shaferlab/tatsu/cBottle-for-ocean-DA/ocean-examples && python train_ocean_diffusion.py'
