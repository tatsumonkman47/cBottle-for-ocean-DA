#!/bin/bash
#
#SBATCH --job-name="ocean-diff-train"
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=60GB
#SBATCH --time=1-00:00:00
#SBATCH --account=torch_pr_432_courant
#SBATCH --output=/home/tm3076/shaferlab/tatsu/cBottle-for-ocean-DA/ocean-examples/logs/train_ocean_diffusion_%j.out
#SBATCH --error=/home/tm3076/shaferlab/tatsu/cBottle-for-ocean-DA/ocean-examples/logs/train_ocean_diffusion_%j.err

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=/home/tm3076/shaferlab/tatsu/cBottle-for-ocean-DA/ocean-examples/logs
LOGFILE="${LOGDIR}/train_ocean_diffusion_${SLURM_LOCALID}_${SLURM_JOB_ID}_${TIMESTAMP}.log"
exec > "$LOGFILE" 2>&1

export WANDB_SILENT=true

singularity exec --nv \
--bind /opt/slurm:/opt/slurm \
--bind /var/run/munge:/var/run/munge \
--bind /etc/ssl/certs/ca-bundle.crt:/etc/ssl/certs/ca-certificates.crt \
--bind /scratch:/scratch \
--overlay /scratch/$USER/singularity_container/overlay-50G-10M.ext3:ro \
/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
/bin/bash -c 'export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt && \
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt && \
export PATH="/opt/slurm/bin:$PATH" && \
unset XLA_FLAGS && unset CUDA_CACHE_PATH && \
source /ext3/env.sh && \
cd /home/tm3076/shaferlab/tatsu/cBottle-for-ocean-DA/ocean-examples && \
python train_ocean_diffusion.py'