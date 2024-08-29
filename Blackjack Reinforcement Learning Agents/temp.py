import numpy as np
import matplotlib.pyplot as plt
from ModifiedBJ_env import ModifiedBlackJack
from monte_carlo_agent import MonteCarloAgent
from sarsa_agent import SarsaLambdaAgent

# Initialize environment and Monte Carlo agent
env = ModifiedBlackJack()
mc_agent = MonteCarloAgent(env)

# Train the Monte Carlo agent
mc_agent.train(num_episodes=90000)

# Get the optimal value function from Monte Carlo
Q_star = mc_agent.Q

# Plot the optimal value function from Monte Carlo
V = mc_agent.get_value_function()
player_sums = range(1, 22)
dealer_shows = range(1, 14)
V_grid = np.zeros((len(player_sums), len(dealer_shows)))

for p_sum in player_sums:
    for d_show in dealer_shows:
        V_grid[p_sum - 1, d_show - 1] = V[(d_show, p_sum)]

X, Y = np.meshgrid(dealer_shows, player_sums)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, V_grid, cmap='viridis')
ax.set_xlabel('Dealer showing')
ax.set_ylabel('Player sum')
ax.set_zlabel('Value')
plt.title("Optimal Value Function from Monte Carlo")
plt.show()