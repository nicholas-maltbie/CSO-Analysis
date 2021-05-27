# Installation Instructions

This project requires python 3.8 ([installation guide](https://wiki.python.org/moin/BeginnersGuide/Download)), anaconda ([installation guide](https://docs.anaconda.com/anaconda/install/index.html)).

The current code is designed to work with CUDA ([installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)) and a NVIDIA GPU as well. This dependency can be avoided by removing gpu references but will make the project exceedingly slow.
* Installing cuDNN ([installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#overview)) can significantly increase the speed of training and operating models as well. 

This project requires the (GNU Scientific Library)[https://packages.debian.org/sid/libgsl-dev] (libgsl-dev) for the [pysbrl](https://github.com/myaooo/pysbrl) library and these are only supported on a Linux system. This library can be installed with the following command:
```bash
apt-get install libgsl-dev
```

Instructions for installing all the required libraries for the python3 and anaconda environments are listed below.

```bash
# Commands to create environment
conda create -n explain-proj python=3.8 pandas numpy scikit-learn numba jupyter jupyterlab seaborn -y

# Activate environment
conda activate explain-proj

# Install ML libraries
# Ensure you have CUDA installed
# Guide can be fairly complex https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
conda install -c conda-forge tensorflow-gpu tensorboard -y
conda install -c anaconda pytables

# Ensure you have libgsl-dev installed
# Install on ubuntu with sudo `apt-get install libgsl-dev`
python3 -m pip install lime shap deeplift git+https://github.com/myaooo/pysbrl

git clone git+https://github.com/rulematrix/rule-matrix-py
cd rule-matrix-py
python3 -m pip install -e .

# To delete environment late, use command
# `conda remove --name explain-proj --all`

```

To verify that this is setup correctly, the project can be launched using jupyter notebook or jupyter lab using the following command:
```bash
jupyter lab --port=8888 --ip=0.0.0.0
```

Running the `Data_Preparation.ipynb` notebook and verifying that all of the imports are setup correctly in the first few cells can verify that all the libraries were setup correctly.

The repository contains generated versions of all the files and steps in the process. 
