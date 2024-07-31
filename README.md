
![image](docs/source/logo.png)
# PyCCE Code Repository
Welcome to the repository, containing source code of **PyCCE** - a Python library for computing qubit dynamics
in the central spin model with cluster-correlation expansion (CCE) method.

### Installation
Run 
`python setup.py install`
in the main folder.

#### Install on ***Perlmutter with GPU support***
Please checkout [install_perlmutter.sh](./examples/mecce_gpu/install_perlmutter.sh) for understanding the dependencies and installation steps.

Run 
```bash
bash install_perlmutter.sh
```
in the `examples/mecce_gpu/` folder.

### Base Units

* Gyromagnetic ratios are given in rad / ms / G.
* Magnetic field in G.
* Timesteps in ms. 
* Distances in A.
* All coupling constants are given in kHz.


### Usage

Usage consists of two steps: preparation of spin bath `BathArray` from `BathCell` and calculations with `Simulator` class.

See `examples` folder for tutorials and scripts of calculations.

#### GPU Usage
See `examples/mecce_gpu` folder for [a demo calculation](./examples/mecce_gpu/mecce.py) and the [script](./examples/mecce_gpu/submit_perlmutter.sh) for submitting job to Perlmutter with 2 GPU nodes (4 GPUs per node).

### Documentation

Full documentation is available online at [Read the Docs](https://pycce.readthedocs.io/en/latest/). 

