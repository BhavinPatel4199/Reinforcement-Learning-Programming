# Blackjack Reinforcement Learning Agents 

- This repository contains implementations of reinforcement learning agents for a customized Blackjack environment. 
- The main agents implemented are Monte Carlo and Sarsa(λ), both of which are trained to optimize their strategies in the Blackjack game.

## Environment : 

### Modified Blackjack Environment : 
  - The "ModifiedBJ_env.py" file contains a custom environment for playing Blackjack, which is used to train the reinforcement learning agents. 
  - The environment is designed to provide the necessary state and reward structure for the agents to learn effective strategies.

## Monte Carlo Agent : 

- The "monte_carlo_agent.py" file implements a Monte Carlo agent, which uses the Monte Carlo method to learn the optimal policy for playing Blackjack. 
- The agent employs an epsilon-greedy strategy for action selection and updates its value function based on the outcomes of entire episodes.

### Key Features :
  - Epsilon-greedy action selection
  - Action-value function (Q-learning)
  - Supports custom initialization and training

## Sarsa(λ) Agent : 

- The `sarsa_agent.py` file contains the implementation of a Sarsa(λ) agent. 
- This agent uses the Sarsa(λ) algorithm, which is a variation of the Sarsa algorithm incorporating eligibility traces, to learn an optimal policy in the Blackjack environment.

### Key Features :
  - Epsilon-greedy action selection
  - Action-value function with eligibility traces
  - Supports custom λ values for training

## Installation : 
- To run the code in this repository, you'll need to install the required dependencies. It's recommended to use a virtual environment.


# Create a virtual environment : 
- python3 -m venv env
- source env/bin/activate 

# On Windows use : 
- `env\Scripts\activate`

## Usage

### To train the Monte Carlo agent:
- python code : 
"""
from monte_carlo_agent import MonteCarloAgent
from ModifiedBJ_env import ModifiedBlackJack
env = ModifiedBlackJack()
mc_agent = MonteCarloAgent(env)
mc_agent.train(num_episodes=90000)
"""

### To train the Sarsa(λ) agent:
- python code :
"""
from sarsa_agent import SarsaLambdaAgent
from ModifiedBJ_env import ModifiedBlackJack

env = ModifiedBlackJack()
sarsa_agent = SarsaLambdaAgent(env)
sarsa_agent.train(num_episodes=90000)
"""

## Visualization : 
- The results of training, such as the optimal value function, can be visualized using the built-in plotting functions:

- python code : 
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# mc_agent is a trained Monte Carlo agent
V = mc_agent.get_value_function()
player_sums = range(1, 22)
dealer_shows = range(1, 14)
V_grid = ...
# Code to generate the plot
plt.show()
"""













