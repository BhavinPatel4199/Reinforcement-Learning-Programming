# Dynamic Programming for Georgian College Parking Optimization

- This repository contains a dynamic programming-based approach to optimize parking space allocation at Georgian College using various algorithms like Policy Evaluation, Policy Iteration, and Value Iteration. 
- The goal is to maximize social welfare by finding the optimal pricing scheme for parking, ensuring that the parking spaces are utilized efficiently while maintaining a balance between full occupancy and availability.

## Overview

- This project is focused on optimizing the parking pricing strategy at Georgian College using dynamic programming techniques. 
- The problem is modeled as a Markov Decision Process (MDP), where the goal is to determine the optimal pricing policy that maximizes social welfare. 
- The repository includes implementations of Policy Evaluation, Policy Iteration, and Value Iteration algorithms.

## Implemented Algorithms

### 1. Policy Evaluation

  - Policy Evaluation is used to determine the value function for a given policy by applying the Bellman equation iteratively. This helps in understanding the effectiveness of the current policy.

### 2. Policy Iteration
  - Policy Iteration alternates between evaluating the current policy and improving it to find a more efficient policy. The process continues until the policy converges to the optimal policy.

### 3. Value Iteration
  - Value Iteration is an algorithm that iteratively applies the Bellman optimality equation to find the optimal value function, which can then be used to derive the optimal policy.

## Environment Setup
  - The environment for this project is defined in the tools.py file. 
  - It includes the class GCParking, which models the parking problem at Georgian College. 
  - This environment provides methods to simulate state transitions and rewards based on different pricing strategies.

- #### Setting Up the Environment
  - Ensure that the tools.py file is in the same directory as your main code. The environment can be initialized as follows:

## Dependencies
  - To run the code in this repository, you will need the following Python libraries:
  - NumPy
  - Matplotlib    
  - Pickle (if required for loading/saving objects)

                
## Results and Visualization
  - The project includes functions to visualize the value functions and policies derived from the algorithms. 
  - The visualizations help in understanding how the parking spaces are valued under different policies and how the policy affects parking utilization.

## Example output:
- Value Function Plot: Shows the expected return for each state under the current policy.
- Policy Plot: Displays the policy as a grid, indicating the action with the highest probability at each state.

## Contributing
- Contributions are welcome! If you have suggestions for improvements or new features, feel free to submit a pull request or open an issue.
