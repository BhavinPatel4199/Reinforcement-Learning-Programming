# Reinforcement-Learning-Programming
### Projects

1. [Dynamic Programming for Georgian College Parking Optimization](./Dynamic%20Programming%20for%20Georgian%20College%20Parking%20Optimization/)
2. [Blackjack Reinforcement Learning Agents](./Blackjack%20Reinforcement%20Learning%20Agents/)
3. [LunarLander Actor-Critic Reinforcement Learning](./LunarLander%20Actor-Critic%20Reinforcement%20Learning/)



You can use these in your documentation or README files.

- This repository contains various projects demonstrating the application of reinforcement learning and dynamic programming techniques to solve real-world problems.

## Projects

### 1. Dynamic Programming for Georgian College Parking Optimization

- This project applies dynamic programming to optimize parking space allocation at Georgian College. The goal is to maximize social welfare by finding the optimal pricing scheme, balancing full occupancy and availability.

- **Overview**: The problem is modeled as a Markov Decision Process (MDP). The project implements Policy Evaluation, Policy Iteration, and Value Iteration algorithms to derive the optimal pricing policy.
- **Implemented Algorithms**:
  - Policy Evaluation
  - Policy Iteration
  - Value Iteration
- **Environment Setup**: Defined in the `tools.py` file, which includes the `GCParking` class for simulating state transitions and rewards.
- **Visualization**: Functions are provided to visualize value functions and policies.

[Detailed Project Documentation](./1.%20Dynamic%20Programming%20for%20Georgian%20College%20Parking%20Optimization)

### 2. Blackjack Reinforcement Learning Agents

This project focuses on developing reinforcement learning agents for a customized Blackjack environment. The main agents implemented are Monte Carlo and Sarsa(λ), both optimized for playing Blackjack.

- **Environment**: The custom environment is defined in `ModifiedBJ_env.py`, designed to support the necessary state and reward structure.
- **Monte Carlo Agent**: Implemented in `monte_carlo_agent.py` using an epsilon-greedy strategy and action-value function (Q-learning).
- **Sarsa(λ) Agent**: Implemented in `sarsa_agent.py` with eligibility traces and support for custom λ values.
- **Visualization**: The project includes code for visualizing the optimal value function.

[Detailed Project Documentation](./"Blackjack Reinforcement Learning Agents")

### 3. LunarLander Actor-Critic Reinforcement Learning

This project demonstrates the application of the Actor-Critic reinforcement learning method to train an agent to play the LunarLander-v2 game.

- **Overview**: 
  - Actor Network: The policy network that selects actions based on the current state.
  - Critic Network: The value network that evaluates the actions.
  - Training: The agent is trained using the Actor-Critic method to maximize cumulative rewards.
  - Recording: The gameplay of the trained agent is recorded as MP4 videos.
- **Files**:
  - `LunarLanderEnv.py`: Defines the environment class.
  - `train.py`: Code for training the agent.
  - `record_video.py`: Code for recording gameplay.
- **Outputs**: Trained models (`actor.pth`, `critic.pth`) and gameplay videos (`lunar_lander.mp4`, `lunar_lander_1.mp4`).

[Detailed Project Documentation](./3.%20LunarLander%20Actor-Critic%20Reinforcement%20Learning)

## Getting Started

- To get started with any of the projects in this repository, clone the repository and follow the instructions provided in the respective project folders.

```bash
git clone https://github.com/krishnapatel1722/Reinforcement-Learning-Programming.git
```
