# Deep Deterministic Policy Gradient (DDPG) Actor Critic 

The repository provides an implementation of a Deep Deterministic Policy Gradient algorithm to solve the Reacher problem.
The agent is supposed to solve the Unity-Reacher problem: a robotic arm shall be trained to move to a moving and remain 
in a moving target location. The agent receives a 33 dimensional input describing the current state and has a four dimensional
continuous action space ranging from -1 to 1. As long as the arm remains in the target location the agent receives a positive reward,
while receiving no reward if the arm is outside the target. The task is episodic with a fixed length of 1000 timesteps and
is considered as solved if the agent receives an average reward of 30 per episode in 100 consecutive episodes.
  
 ![Alt text](reacher.png?raw=true "Title")


## Installation
Clone the github repo
- git clone https://github.com/ChristianH1984/pg_continuous_control.git
- cd dqn_banana
- conda env create --name pg_continuous_control --file=environment.yml
- activate pg_continuous_control
- open Report.ipynb and have fun :-)