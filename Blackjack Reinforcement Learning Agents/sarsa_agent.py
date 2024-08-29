import numpy as np
import pandas as pd
from collections import defaultdict

class SarsaLambdaAgent:
    """
    A Sarsa(λ) agent for training on a reinforcement learning environment.

    Attributes:
        env: The environment on which the agent is trained.
        N0: The initial value for the epsilon-greedy exploration strategy.
        lambdas: A list of λ values to be used during training.
        Q_star: Placeholder for true Q-values from Monte Carlo.
        Q: The action-value function.
        results: A list to store training results.
    """

    def __init__(self, env, N0=100, lambdas=None):
        """
        Initializes the SarsaLambdaAgent with the given environment and parameters.

        Args:
            env: The environment on which the agent is trained.
            N0: The initial value for the epsilon-greedy exploration strategy. Default is 100.
            lambdas: A list of λ values to be used during training. If None, defaults to np.arange(0, 1.1, 0.1).
        """
        
        self.env = env
        self.N0 = N0
        self.lambdas = lambdas if lambdas is not None else np.arange(0, 1.1, 0.1)
        self.Q_star = None  # Placeholder for true Q-values from Monte Carlo
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.results = []

    def epsilon_greedy_policy(self, state, N_state):
        """
        Selects an action using an epsilon-greedy policy based on the current state.

        Args:
            state: The current state of the environment.
            N_state: A dictionary tracking the number of times each state has been visited.

        Returns:
            The action selected according to the epsilon-greedy policy.
        """

        epsilon = self.N0 / (self.N0 + N_state[state])
        if np.random.rand() < epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])

    def train(self, num_episodes=1000):
        """
        Trains the agent using the Sarsa(λ) algorithm for a specified number of episodes.

        Args:
            num_episodes: The number of episodes for training.

        Returns:
            A list of tuples, each containing a λ value and the corresponding mean squared errors over episodes.
        """

        all_errors = []
        for lamb in self.lambdas:
            self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
            N = defaultdict(lambda: np.zeros(self.env.action_space.n))
            N_state = defaultdict(int)
            lamb_errors = []

            for episode_num in range(num_episodes):
                E = defaultdict(lambda: np.zeros(self.env.action_space.n))  # Eligibility traces
                state = self.env.reset()
                action = self.epsilon_greedy_policy(state, N_state)
                done = False
                G = 0

                while not done:
                    next_state, reward, done = self.env.step(state, action)
                    G = reward + G
                    next_action = self.epsilon_greedy_policy(next_state, N_state)
                    delta = reward + (0 if done else self.Q[next_state][next_action]) - self.Q[state][action]
                    E[state][action] += 1

                    for s in self.Q:
                        for a in range(self.env.action_space.n):
                            self.Q[s][a] += (1 / (N[s][a] + 1)) * delta * E[s][a]
                            if not done:
                                E[s][a] *= lamb

                    if not done:
                        state, action = next_state, next_action

                N_state[state] += 1
                N[state][action] += 1
                mse = self.compute_mse()
                lamb_errors.append(mse)
                self.results.append({'episode': episode_num, 'lambda': lamb, 'MSE': mse})

            all_errors.append((lamb, lamb_errors))
        return all_errors

    def compute_mse(self):
        """
        Computes the mean squared error (MSE) of the current Q-values compared to the true Q-values (Q_star).

        Returns:
            The mean squared error between the current Q-values and the true Q-values.
        
        Raises:
            ValueError: If Q_star (true Q-values) is not set.
        """

        if self.Q_star is None:
            raise ValueError("Q_star (true Q-values) must be set before computing MSE.")
        mse = 0
        count = 0
        for state in self.Q_star:
            for action in range(self.env.action_space.n):
                mse += (self.Q[state][action] - self.Q_star[state][action]) ** 2
                count += 1
        mse /= count
        return mse

    def set_Q_star(self, Q_star):
        """
        Sets the true Q-values (Q_star) to be used for computing the MSE.

        Args:
            Q_star: The true Q-values obtained from the Monte Carlo agent.
        """

        self.Q_star = Q_star

    def get_results(self):
        """
        Retrieves the training results as a pandas DataFrame.

        Returns:
            A DataFrame containing the training results.
        """
        
        return pd.DataFrame(self.results)
