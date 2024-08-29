import gym
from gym import spaces
import numpy as np

class ModifiedBlackJack(gym.Env):
    """
    A custom environment for a modified version of Blackjack using the OpenAI Gym framework.

    Attributes:
        action_space: The action space, which includes two possible actions: Stick (0) or Hit (1).
        observation_space: The observation space, which includes the player's sum and the dealer's visible card.
    """
    
    def __init__(self):
        """
        Initializes the ModifiedBlackJack environment.
        """
        
        super(ModifiedBlackJack, self).__init__()
        self.action_space = spaces.Discrete(2)  # Stick (0) or Hit (1)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),  # Player's sum (0 to 31)
            spaces.Discrete(13),  # Dealer's visible card (1 to 13)
        ))
        self.reset()

    def reset(self):
        """
        Resets the environment to an initial state and returns an initial observation.

        Returns:
            A tuple representing the initial observation: the dealer's visible card and the player's sum and done variable.
        """
        
        self.player_cards = [(self.draw_black_card(), 'black')]
        self.dealer_cards = [(self.draw_black_card(), 'black')]
        self.player_sum = self.calculate_sum(self.player_cards)
        self.dealer_sum = self.calculate_sum(self.dealer_cards)
        self.done = False
        return (self.dealer_cards[0][0], self.player_sum)

    def step(self, state, action):
        """
        Executes one time step within the environment based on the given action.

        Args:
            state: The current state of the environment.
            action: The action to be taken (0 for Stick, 1 for Hit).

        Returns:
            A tuple containing the next state, the reward, and a boolean indicating if the episode is done.
        """
        
        dealer_card, player_sum = state

        if action == 1:  # Hit
            card_value, card_color = self.draw_card()
            self.player_cards.append((card_value, card_color))
            self.player_sum = self.calculate_sum(self.player_cards)
            if self.player_sum > 21 or self.player_sum < 1:
                self.done = True
                return (dealer_card, self.player_sum), -1, True
            else:
                return (dealer_card, self.player_sum), 0, False

        elif action == 0:  # Stick
            self.done = True
            while self.dealer_sum < 17:
                card_value, card_color = self.draw_card()
                self.dealer_cards.append((card_value, card_color))
                self.dealer_sum = self.calculate_sum(self.dealer_cards)
                if self.dealer_sum > 21 or self.dealer_sum < 1:
                    return (dealer_card, self.player_sum), 1, True

            if self.player_sum > self.dealer_sum:
                return (dealer_card, self.player_sum), 1, True
            elif self.player_sum < self.dealer_sum:
                return (dealer_card, self.player_sum), -1, True
            else:
                return (dealer_card, self.player_sum), 0, True
            
    def _get_obs(self):
        """
        Returns the current observation of the environment.

        Returns:
            A tuple representing the current observation: the player's sum and the dealer's sum.
        """

        return (self.player_sum, self.dealer_sum)

    def draw_card(self):
        """
        Draws a random card from the deck.

        Returns:
            A tuple representing the card value and its color (red or black).
        """
        
        value = np.random.randint(1, 14)
        color = "red" if np.random.rand() < 1/3 else "black"
        return value, color

    def draw_black_card(self):
        """
        Draws a random black card from the deck.

        Returns:
            The value of the black card.
        """

        return np.random.randint(1, 14)

    def calculate_sum(self, cards):
        """
        Calculates the sum of the given cards, considering their colors.

        Args:
            cards: A list of tuples representing the cards and their colors.

        Returns:
            The calculated sum of the cards.
        """
        
        total = 0
        for value, color in cards:
            if color == "black":
                total += value
            else:
                total -= value
        return total

    def render(self, mode='human'):
        """
        Renders the current state of the environment.

        Args:
            mode: The mode in which to render the environment. Default is 'human'.
        """
        
        print("Player's cards: ", [(value, color) for value, color in self.player_cards], "| Player's sum: ", self.player_sum)
        print("Dealer's cards: ", [(value, color) for value, color in self.dealer_cards], "| Dealer's sum: ", self.dealer_sum)
        print()
        if self.done:
            if self.player_sum > 21 or self.player_sum < 1:
                print("Player went bust! Dealer wins.")
            elif self.dealer_sum > 21 or self.dealer_sum < 1:
                print("Dealer went bust! Player wins.")
            elif self.player_sum > self.dealer_sum:
                print("Player wins!")
            elif self.player_sum < self.dealer_sum:
                print("Dealer wins!")
            else:
                print("It's a draw!")

    def close(self):
        pass

# Usage
env = ModifiedBlackJack()
obs = env.reset()
print("Initial observation:", obs)
for _ in range(10):
    action = env.action_space.sample()  # Random action: 0 or 1
    obs, reward, done = env.step(obs, action)
    #print(f'action: {action}')
    #print(f'reward: {reward}') 
    env.render()
    if done:
        break
