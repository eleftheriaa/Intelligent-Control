# **Implementing Soft Actor-Critic (SAC) from Scratch**
*Topics:* Actor-Critic, Maximum Entropy RL, Continuous Control

*Description:* Implement the SAC algorithm from scratch using PyTorch or JAX and apply
it to robotics tasks. Analyze the impact of entropy regularization, target networks, and Qfunction stability.


# **Setup and Running Instructions**

## 1.First, clone this repository to your local machine:
#### i. Open a terminal and run:

    git clone https://github.com/eleftheriaa/Intelligent-Control.git

    cd Intelligent-Control
    

To run the SAC algorithm implementation successfully, follow these steps:

## 2. Create a Conda Environment (Recommended)


####   Using a virtual environment ensures that dependencies do not interfere with other projects.
  

####   i. Open a terminal and run:
  
     conda create -n sac_env python=3.10 -y
     
     conda activate sac_env

## 3.  Install Required Libraries


####   Once the environment is activated, install the following required packages:
  
     pip install numpy torch gymnasium gymnasium-robotics tensorboard

## 4.  Run the Program

#### Before running training, make sure to install the repository as a package and run the experiment script once to initialize logs and directories:
    pip install -e .
    python experiments/main.py


####   Make sure you are in the directory that contains the main Python file (e.g., main.py) and run:
  
     python main.py


####   This will begin training the agent on the selected environment (default: "PointMaze_UMaze-v3")
  

####   TensorBoard logs will be saved in the runs/ directory.

## 5.  Visualize Training with TensorBoard


####   To monitor the training in real time:
  

####   i. Open a new terminal in visual studio
  

####   ii. Activate the same conda environment (conda activate sac_env)*
  

####   iii.Run:
    
     tensorboard --logdir=runs/
   

####   iv.Then, open your browser and go to suggested link

***step ii. is not necessary if you are running inside visual studio**




