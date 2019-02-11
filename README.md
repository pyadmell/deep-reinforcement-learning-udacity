# Udacity Deep Reinforcement Learning Nanodegree Projects
 
This repository contains the Udacity's Deep Reinforcement Learning Nanodegree projects.

## Table of Contents

1. [Project Details](#project-details)
    1. [Collaboration and Competition](#collaboration-and-competition)
        1. [State-Action Represenation](#state-action-represenation)
        1. [Reward](#reward)
1. [Getting Started & Dependencies](#getting-started-and-dependencies)
1. [Instructions](#instructions)

---

## Project Details

![tennis-env](./doc/gif/tennis-env.gif)

### [Collaboration and Competition](./p3_collab-compet/Report.md)

The Collaboration and Competition project is based on the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)
environment from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents).
The environment simulates a two-player table tennis game, where agents control rackets to bounce ball over a net.
The goal in this project is to implement and train an agent to control the players to keep the ball in play. Each player receives a reward of +0.1 if it hits the ball over the net, 
and -0.1 if it lets the ball hit the ground or go out of bounds.
The task is episodic, and the environment is considered solved when the agent reaches an average score of +0.5 over 100 consecutive episodes. 

#### State-Action Represenation

- Observation space type: continuous
    - Observation space size (per agent): 8, corresponding to:
        - position and velocity of ball and racket
- Action space type: discrete
    - Action space size (per agent): 2 (continuous), corresponding to:
        - movement toward net or away from net, and jumping

#### Reward

- +0.1 if the player hits the ball over net.
- -0.1 if the player let ball hit the ground, or hit ball out of bounds.

## Getting Started and Dependencies

This project depends on Tennis environment and PyTorch along with some other Python packages. Follow the instructions below to install the dependencies and set up the python environment:

0. Download the Reacher environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

1. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 
2. Download and install [miniconda3](https://conda.io/miniconda.html).
3. Create the miniconda environment:
```bash
conda env create -f environment.yml
```
4. Verify the `drlnd` environment:Instructions
```bahs
conda info --envs
```
5. Clean up downloaded packages:
```bash
conda clean -tp
```
6. Activate `drlnd` conda environment:
```bash
conda activate drlnd
```
7. Clone the [Udacity's deep-reinforcement-learning repository](https://github.com/udacity/deep-reinforcement-learning), and navigate to the repository folder to install the additional dependencies including the ML-Agents toolkit, and a few more Python packages required for this project:
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning
pip -q install ./python
```
8. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
9. Run jupyter notebook.
```bash
jupyter-notebook .
```
10. Before running the jupyter notebooks, make sure the kernel is set to `drlnd`. If not, change the environment by using the drop-down `Kernel` menu. 

## Instructions

- Project folder: [p3_collab-compet/](./p3_collab-compet/)
- Project files:
    - [Report.md](./p3_collab-compet/Report.md): project report and solution
    - [actor_critic.py](./p3_collab-compet/actor_critic.py): the neural network model architecture
    - [utils.py](./p3_collab-compet/utils.py): utilities for creating network
    - [Tennis-PPO.ipynb](./p3_collab-compet/Tennis-PPO.ipynb): PPO implementation, training, and evaluation
    - [ppo_128x128_a64_c64_207e.pth](./p3_collab-compet/ppo_128x128_a64_c64_207e.pth): saved model weights
- Refer to [/p3_collab-compet](./p3_collab-compet) folder for the solution implementation and [report](./p3_collab-compet/Report.md). 
