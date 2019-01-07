# Udacity Deep Reinforcement Learning Nanodegree Projects
 
This repository contains the Udacity's Deep Reinforcement Learning Nanodegree projects.

## Table of Contents

1. [Project Details](#project-details)
    1. [Navigation](#navigation)
    1. [Continuous Control](#continuous-control)
    1. [Collaboration and Competition](#collaboration-and-competition)
1. [Getting Started & Dependencies](#getting-started-and-dependencies)
1. [Instructions](#instructions)

---

## Project Details

All of the projects are based on simulation environments from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents). The projects include:

### [Navigation](./p1_navigation/Report.ipynb)

The goal in this project is to implement and train a DQN agent to collect yellow bananas while avoiding blue bananas. Refer to [/p1_navigation](./p1_navigation) folder for the project description and [report](./p1_navigation/Report.ipynb). 

![banana-collector](./doc/gif/banana-collector.gif)

### Continuous Control

TBD

### Collaboration and Competition

TBD


## Getting Started and Dependencies

Ffollow the instructions below to install the dependencies and set up the python environment:

1. Download and install [miniconda3](https://conda.io/miniconda.html).
2. Create the miniconda environment:
```bash
conda env create -f environment.yml
```
3. Verify the `drlnd` environment:Instructions
```bahs
conda info --envs
```
4. Clean up downloaded packages:
```bash
conda clean -tp
```
5. Activate `drlnd` conda environment:
```bash
conda activate drlnd
```
6. Clone the [Udacity's deep-reinforcement-learning repository](https://github.com/udacity/deep-reinforcement-learning), and navigate to the repository folder to install the additional dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning
pip -q install ./python
```
7. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
8. Run jupyter notebook.
```bash
jupyter-notebook .
```
9. Before running the jupyter notebooks, make sure the kernel is set to `drlnd`. If not, change the environment by using the drop-down `Kernel` menu. 

## Instructions

- Navigation:
    - Project folder: [p1_navigation](./p1_navigation)
    - Project files:
        - [Report.ipynb](./p1_navigation/Report.ipynb): project report and solution
        - [dqn_agent.py](./p1_navigation/dqn_agent.py): dqn implementation
        - [model.py](./p1_navigation/model.py): the neural network model architecture
        - [banana-32-32-checkpoint.pth](./p1_navigation/banana-32-32-checkpoint.pth): saved model weights
        - [training_plot.png](./p1_navigation/training_plot.png): the training plot showing reward per episode
    - Follow [Report.ipynb](./p1_navigation/Report.ipynb) for further details.
