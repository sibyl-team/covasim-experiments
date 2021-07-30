# Testing covasim 

This repository contains the notebook used for the testing

Install first covasim -> https://github.com/sibyl-team/covasim

Then install additional modules -> https://github.com/sibyl-team/covasibyl

## Python environments

Since covasim requires packages with are not available in the Conda repositories,
you should create a new virtual environment to mantain a clear separation.

### Instructions for dummies

1. Activate your conda environment

2. Go to the folder in which you want to make the new virtual environment ("folder"). Type 
    ```
    python -m venv "folder"/covasim
    ```
    and it will create a new environment in `folder/covasim` (replace placeholders!)

3. Activate the new environment (you don't really need to activate conda environment first now)
    ```
    source "folder"/covasim/bin/activate
    ```
4. Navigate to the folder containing the codes, and run `pip install .` to install the package in the new environment,
or `pip install -e .` if you want to be able to change the source on the run and make new functionality

5. You should install also `ipykernel` and install the new kernel if you want to use the Jupyter notebook.
