import gymnasium as gym
import numpy as np

class LunarLanderEnv:
    def __init__(self):
        self.env = gym.make('LunarLander-v2', render_mode="rgb_array")
        self._state_space = self.env.observation_space.shape[0]
        self._action_space = self.env.action_space.n
        self.__S = list(range(self._state_space))
        self.__A = list(range(self._action_space))

    def reset(self):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        return next_state, reward, done, truncated, info

    def render(self):
        self.env.render()

    def transitions(self, s, a):
        next_state, reward, done, truncated, info = self.step(a)
        return np.array([[next_state, reward, self.p(next_state, reward, s, a)]])

    def support(self, s, a):
        next_state, reward, done, truncated, info = self.step(a)
        return [(next_state, reward)]

    def p(self, s_, r, s, a):
        if r != self.reward(s, s_):
            return 0
        else:
            center = s + self._action_space * (1 - a / self._action_space)
            emphasis = np.exp(-abs(np.arange(2 * self._state_space) - center) / 5)
            if s_ == self._state_space:
                return sum(emphasis[s_:]) / sum(emphasis)
            return emphasis[s_] / sum(emphasis)

    def reward(self, s, s_):
        return self.state_reward(s) + self.state_reward(s_)

    def state_reward(self, s):
        return s

    def random_state(self):
        return np.random.randint(self._action_space)

    @property
    def A(self):
        return list(self.__A)

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def S(self):
        return list(self.__S)

class Transitions(list):
    def __init__(self, transitions):
        self.__transitions = transitions
        super().__init__(transitions)

    def __repr__(self):
        repr = '{:<14} {:<10} {:<10}'.format('Next State', 'Reward', 'Probability')
        repr += '\n'
        for i, (s, r, p) in enumerate(self.__transitions):
            repr += '{:<14} {:<10} {:<10}'.format(s, round(r, 2), round(p, 2))
            if i != len(self.__transitions) - 1:
                repr += '\n'
        return repr

# Sample usage
if __name__ == "__main__":
    env = LunarLanderEnv()

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.random.choice(env.A)
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = next_state

    print(f'Total Reward: {total_reward}')
    env.env.close()  # Close the rendering window properly
