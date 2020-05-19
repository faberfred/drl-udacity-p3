[//]: # (Image References)

# Collaboration and Competition - Readme

### Introduction & goal of the project

This is the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) by Udacity.
The objective of this project is to train two agents to control rackets to bounce a ball over a net. It's a Unity environment provided by Udacity.  

![Training results](./tennis.png  "Training results")

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Given this information, the agent has to learn how to best select actions. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

To solve the environment, the agents must get an average score of +0.5 or better over 100 consecutive episodes.

A reward of +0.1 is provided if an agent hits the ball over the net.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

### Settings

**Implementation:** The project is implemented in `python 3.6` and within a `jupyter notebook`. 

**`Pytorch 1.4.0`** has been used with **`CUDAToolkit 10.1`**.
The operating system is `ubuntu 18.04`.
Unity 2019.3 has been installed on the computer.
The Unity environment is `Tennis_Linux/Tennis.x86_64` and has been provided by Udacity. It cab ne downloaded [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)

**Train the agent by executing the whole jupyter notebook from the beginning to the end**