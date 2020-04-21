# Deep Reinforcement Learning project: reacher
## Project details

In this project, an agent is trained to solve an environment in which a double-jointed arm can move to target locations.

![reacher environment](images/reacher.gif "Reacher environment")

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to  position, rotation, velocity, and angular velocities of the arm. Each  action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number  between -1 and 1.

In order to solve the environment, the  agent must get an average score of +30 over 100 consecutive episodes.

There exist two different versions of this environment. In the first one, there is only one agent, while in the second one there are 20 agents. The environment solved has just one agent to train.

## Getting started

1. Follow the instructions given [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install all the dependencies.

2. Download the environment for your OS:

   * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip) for one agent and [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) for 20 agents.

   * Mac OS: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip) for one agent and [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip) for 20 agents.
   * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip) and [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip) for 20 agents.
   * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip) and [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip) for 20 agents.

3. Place the file in the root of the folder and unzip it.

4. Run it! Consider to change the cell at point 2 of the notebook to match with your folder `env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')`.

## Instructions

If you want to train the agent, run the cells from the point 1 to the point 6 at [DDPG.ipynb](DDPG.ipynb). If you just want to execute the trained agent, run the cells from the point 1 to the point 4 plus the 7 at the notebook, since they load the weights of the networks ([checkpoint_actor.pth](checkpoint_actor.pth) for the actor and [checkpoint_critic.pth](checkpoint_critic.pth) for the critic) for the trained agent. After running whatever you want, you can close the environment by running the cell at point 8.

Description of the implementation is in [Report.md](Report.md), but for more technical details, see the code at the notebook provided before.