# LunarLander Actor-Critic Reinforcement Learning

- This project demonstrates how to train an agent to play the LunarLander-v2 game using the Actor-Critic reinforcement learning method. 
- The project includes code for defining the environment, neural networks for the actor and critic, training the model, and recording gameplay videos to visualize the agent's performance.

## Overview

- **Actor Network**: The policy network that decides the action to take based on the current state.
- **Critic Network**: The value network that evaluates the action taken by the actor.
- **Training**: The agent is trained using the Actor-Critic method to maximize the cumulative reward.
- **Recording**: After training, the agent's gameplay is recorded and saved as an MP4 video.

## Files

- `LunarLanderEnv.py`: Defines the environment class.
- `train.py`: Contains the code for training the agent.
- `record_video.py`: Contains the code for recording the agent's gameplay.
- `actor.pth`: The trained actor model.
- `critic.pth`: The trained critic model.
- `lunar_lander.mp4`: Video of the trained agent playing the LunarLander-v2 game.
- `lunar_lander_1.mp4`: Video of the trained agent playing a different environment.

## Requirements

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```


## Running the Project

- To train the model, run:
```bash
python train.py
```
This will generate actor.pth and critic.pth files which store the trained models.

## Record Gameplay
  - To record the agent's gameplay after training, run:

```bash
python record_video.py
```
- This will save a video file showcasing the agent's performance.

## Displaying Performance
- To display how the Actor-Critic Neural Network performs in different environments, you can view the following videos:
1. LunarLander-v2 Environment
2. Alternative Environment

## Contributing
- Contributions are welcome! If you have any suggestions or improvements, feel free to submit a pull request.
