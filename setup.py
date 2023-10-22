import setuptools
from setuptools import setup
import subprocess

def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        for line in output.split('\n'):
            if 'release' in line:
                # Change the parsing method to account for the example output
                return line.split('release')[-1].split(',')[0].strip().split(' ')[-1]
    except Exception as e:
        print(f"Failed to detect CUDA version with error: {e}. Assuming no CUDA is installed.")
        return None

def get_cupy_version(cuda_version):
    major, minor = map(int, cuda_version.split('.'))
    
    if major == 10 and minor == 2:
        return 'cupy-cuda102'
    elif major == 11:
        if 0 <= minor <= 1:
            return f'cupy-cuda11{minor}'
        elif 2 <= minor <= 8:
            return 'cupy-cuda11x'
    elif major == 12:
        return 'cupy-cuda12x'
    
    return None

cuda_version = get_cuda_version()
if cuda_version:
    cupy_install = get_cupy_version(cuda_version)
    if not cupy_install:
        print(f"CuPy version for CUDA {cuda_version} not defined. Skipping CuPy installation.")
else:
    cupy_install = None

install_requires = [
'numpy', 'scipy', 'ase', 'pandas'
]

if cupy_install:
    install_requires.append(cupy_install)

setup(
    name='gpu4pycce',
    version='0.0.1',
    url='',
    license='',
    author='Jiawei Zhan',
    author_email='jiaweiz@uchicago.edu',
    description='A plugin to use Nvidia GPU in PyCCE package',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    package_data={'': ['bath/*.txt']},
)
