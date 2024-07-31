#!/bin/bash

module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80
module load python
conda create -n gpu4pycce python -y
conda activate gpu4pycce

MPICC="cc -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
conda install -y numpy==1.26.4 scipy==1.13.0 matplotlib
pip3 install tqdm numba ase pandas torch

