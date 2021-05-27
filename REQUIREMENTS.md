# Project Requirements

In order to run the project, a few requirements must be met in order to operate the project.

## Hardware Requirements

This project requires a system with a nvidia GPU and enough ram to load and operate the dataset (4GB should be enough). In addition, downloading the dataset, libraries, and processing the results may require up to 16GB of disk space in total. 

## Operating System Requirements

This project has a library that is only supported on the Linux operation system. If the RuleMatrix part of explainability analysis is skipped, the rest of the project can be run on any operating system. 

## Software Requirements

This project requires python 3.8 ([installation guide](https://wiki.python.org/moin/BeginnersGuide/Download)), anaconda ([installation guide](https://docs.anaconda.com/anaconda/install/index.html)) as well as a set of libraries from these environments.

The list of all the installed libraries and their versions can be found in the `pip-requirements.txt` and `conda-requirements.txt` files in the repository. The commands to install the most up to date versions of these libraries and all other packages are discussed further in the [INSTALL](INSTALL.md) file under the section Setup Instructions.

Python version 3.8 is used as it is the only version where all libraries and dependencies have compatible versions. 
