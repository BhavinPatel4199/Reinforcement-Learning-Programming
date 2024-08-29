import numpy as np
import pandas as pd
from collections import defaultdict

class MonteCarloAgent:
    """
    A Monte Carlo agent for training on a reinforcement learning environment.

    Attributes:
        env: The environment on which the agent is trained.
        N0: The initial value for the epsilon-greedy exploration strategy.
        Q: The action-value function.
        N: A dictionary tracking the number of times each state-action pair has been visited.
        N_state: A dictionary tracking the number of times each state has been visited.
        results: A list to store training results.
    """

    def __init__(self, env, N0=100):
        """
        Initializes the MonteCarloAgent with the given environment and parameters.

        Args:
            env: The environment on which the agent is trained.
            N0: The initial value for the epsilon-greedy exploration strategy. Default is 100.
        """
        
        self.env = env
        self.N0 = N0
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N_state = defaultdict(int)
        self.results = []

    def epsilon_greedy_policy(self, state):
        """
        Selects an action using an epsilon-greedy policy based on the current state.
        The policy is presented in the question.

        Args:
            state: The current state of the environment.

        Returns:
            The action selected according to the epsilon-greedy policy.
        """

        epsilon = self.N0 / (self.N0 + self.N_state[state])
        if np.random.rand() < epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])

    def generate_episode(self):
        """
        Generates an episode by interacting with the environment using the epsilon-greedy policy.

        Returns:
            A list of tuples representing the episode. Each tuple contains (state, action, reward).
        """
        
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            action = self.epsilon_greedy_policy(state)
            next_state, reward, done = self.env.step(state, action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def train(self, num_episodes=1000):
        """
        Trains the agent using the Monte Carlo method for a specified number of episodes.

        Args:
            num_episodes: The number of episodes for training.
        """

        for episode_num in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            states_visited = set()
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + G
                states_visited.add(state)
                self.N_state[state] += 1
                self.N[state][action] += 1
                alpha = 1 / self.N[state][action]
                self.Q[state][action] += alpha * (G - self.Q[state][action])
            self.results.append({'episode': episode_num, 'accumulated_reward': G, 'states_visited': len(states_visited)})

    def get_value_function(self):
        """
        Computes the value function based on the current Q-values.

        Returns:
            A dictionary representing the value function, where the keys are states and the values are the maximum action-value for each state.
        """

        V = defaultdict(float)
        for state, actions in self.Q.items():
            V[state] = np.max(actions)
        return V

    def get_results(self):
        """
        Retrieves the training results as a pandas DataFrame.

        Returns:
            A DataFrame containing the training results.
        """
        
        return pd.DataFrame(self.results)
