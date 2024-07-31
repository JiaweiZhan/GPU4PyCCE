
![image](docs/source/logo.png)
# PyCCE Code Repository
Welcome to the repository, containing source code of **PyCCE** - a Python library for computing qubit dynamics
in the central spin model with cluster-correlation expansion (CCE) method.

### Installation
Run 
`python setup.py install`
in the main folder.

### Base Units

* Gyromagnetic ratios are given in rad / ms / G.
* Magnetic field in G.
* Timesteps in ms. 
* Distances in A.
* All coupling constants are given in kHz.


### Usage

Usage consists of two steps: preparation of spin bath `BathArray` from `BathCell` and calculations with `Simulator` class.

See `examples` folder for tutorials and scripts of calculations.

### Documentation

Full documentation is available online at [Read the Docs](https://pycce.readthedocs.io/en/latest/). 

## GPU support
### Key Features
- Accelerate the **ME-CCE** method utilizing NVIDIA GPUs.
- Enable ***multi-GPU*** and ***multi-node*** computations through GPU-aware MPI.

### Installation on ***Perlmutter with GPU Support***
For detailed dependencies and installation instructions, refer to [install_perlmutter.sh](./examples/mecce_gpu/install_perlmutter.sh)

To set up the environment:

1. Execute the installation script::
```bash
bash ./examples/mecce_gpu/install_perlmutter.sh
```
This will create the `gpu4pycce` Conda environment and install the necessary dependencies.

2. Temporarily add the main directory to PYTHONPATH:
```bash
CURRENT_PATH=`pwd`
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"
```
Alternatively, you can make this change permanent by modifying your `~/.bashrc`:
```bash
echo 'export PYTHONPATH="${PYTHONPATH}:{abs path of the current repo}"' >> ~/.bashrc
source ~/.bashrc
```

### GPU Usage
Explore the `examples/mecce_gpu` directory for a [demonstration calculation](./examples/mecce_gpu/mecce.py) and a [submission script](./examples/mecce_gpu/submit_perlmutter.sh) designed for running jobs on Perlmutter using 2 GPU nodes (4 GPUs per node).

### Performance Benchmarks
GPU4PyCCE demonstrates substantial performance improvements when utilizing an NVIDIA A100 GPU (9.7 TFLOPS on FP64) compared to PyCCE running on a single-core AMD EPYC 7763 (Milan) CPU (39.2 GFLOPS per core). The following table outlines the speedup achieved in the [mecce example](./examples/mecce_gpu/) by leveraging GPU acceleration:

| order               |   speedup |
|:--------------------|----------:|
| 1                   |      5x    |
| 2                   |     56x    |
| 3                   |    250x    |
