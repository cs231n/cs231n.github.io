---
layout: page
title: Setup Instructions
permalink: /setup-instructions/
---
- [Working remotely on Google Cloud](#working-remotely-on-google-cloud)
- [Working locally on your machine](#working-locally-on-your-machine)
  - [Anaconda virtual environment](#anaconda-virtual-environment)
  - [Python venv](#python-venv)
  - [Installing packages](#installing-packages)

You can work on the assignment in one of two ways: **locally** on your own machine, or **remotely** on a Google Cloud virtual machine (VM).

### Working remotely on Google Cloud
As part of this course, you can use Google Cloud for your assignments. We recommend this route for anyone who is having trouble with installation set-up, or if you would like to use better CPU/GPU resources than you may have locally. Please see the set-up tutorial [here](https://github.com/cs231n/gcloud/) for more details.

**Note:** after following these instructions, you may skip the remaining sections.

### Working locally on your machine
If you wish to work locally, you should use a virtual environment. You can install one via Anaconda (recommended) or via Python's native `venv` module. Ensure you are using Python 3.7 as **we are no longer supporting Python 2**.

#### Anaconda virtual environment
We strongly recommend using the free [Anaconda Python distribution](https://www.anaconda.com/download/), which provides an easy way for you to handle package dependencies. Please be sure to download the Python 3 version, which currently installs Python 3.7. The neat thing about Anaconda is that it ships with [MKL optimizations](https://docs.anaconda.com/mkl-optimizations/) by default, which means your `numpy` and `scipy` code benefit from significant speed-ups without having to change a single line of code.

Once you have Anaconda installed, it makes sense to create a virtual environment for the course. If you choose not to use a virtual environment (strongly not recommended!), it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment called `cs231n`, run the following in your terminal:

```bash
conda create -n cs231n python=3.7
```

To activate and enter the environment, run `conda activate cs231n`. To deactivate the environment, either run `conda deactivate cs231n` or exit the terminal. Note that every time you want to work on the assignment, you should rerun `conda activate cs231n`.

You may refer to [this page](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more detailed instructions on managing virtual environments with Anaconda.

**Note:** If you've chosen to go the Anaconda route, you can safely skip the next section and move straight to [Installing Packages](#installing-packages).

<a name='venv'></a>
#### Python venv

As of 3.3, Python natively ships with a lightweight virtual environment module called [venv](https://docs.python.org/3/library/venv.html). Each virtual environment packages its own independent set of installed Python packages that are isolated from system-wide Python packages and runs a Python version that matches that of the binary that was used to create it. To set up your `cs231` venv for the course, run the following:

```bash
# create a virtual environment called cs231n
# that will use version 3.7 of Python
python3.7 -m venv cs231n
source cs231n/bin/activate  # activate the virtual env

# sanity check that the path to the python
# binary matches that of the virtual env.
which python
# for example, on my machine, this prints
# $ '/Users/kevin/cs231n/bin/python'
```

<a name='packages'></a>
#### Installing packages

Once you've **setup** and **activated** your virtual environment (via `conda` or `venv`), you should install the libraries needed to run the assignments using `pip`. To do so, run:

```bash
# again, ensure your virtual env has been activated
# before running the commands below
cd assignment1  # cd to the assignment directory
pip install -r requirements.txt  # install assignment dependencies
# work on the assignment for a while ...
deactivate  # deactivate the virtual env
```
