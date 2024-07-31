#!/bin/bash

#SBATCH --job-name=pycce-mecce
#SBATCH --time=00:30:00
#SBATCH --account=<project_name>
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32

module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python
conda activate gpu4pycce

export OMP_NUM_THREADS=1
export SLURM_CPU_BIND=cores
export MPICH_GPU_SUPPORT_ENABLED=1
export ROMIO_FSTYPE_FORCE="ufs:"

srun -n 8 python3 -u mecce.py
