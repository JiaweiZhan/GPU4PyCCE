
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
### Install on ***Perlmutter with GPU support***
Please checkout [install_perlmutter.sh](./examples/mecce_gpu/install_perlmutter.sh) for understanding the dependencies and installation steps.

1. Run 
```bash
bash ./examples/mecce_gpu/install_perlmutter.sh
```
in the main folder.

2. Add the main folder to the `PYTHONPATH`
```bash
CURRENT_PATH=`pwd`
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"
```

### GPU Usage
See `examples/mecce_gpu` folder for [a demo calculation](./examples/mecce_gpu/mecce.py) and the [script](./examples/mecce_gpu/submit_perlmutter.sh) for submitting job to Perlmutter with 2 GPU nodes (4 GPUs per node).
