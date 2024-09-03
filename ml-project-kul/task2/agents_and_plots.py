# -*- coding: utf-8 -*-
"""Dependencies"""

!pip install open_spiel
!pip install mpltern

"""Imports"""

import numpy as np
import matplotlib.pyplot as plt
import pyspiel
import mpltern

"""## Epsilon Greedy Q Learning Agent"""

class EpsilonGreedyQLearningAgent:
    def __init__(self, num_states, num_actions, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.num_actions = num_actions
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = np.zeros((num_states, num_actions))

    def name(self):
        return 'epsilon-greedy Q-learning'

    def set_q_table(self, q_table):
        self.q_table = q_table

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Random action
        else:
            return np.argmax(self.q_table[state])  # Greedy action

    def update_q_table(self, state, action, reward, next_state):
        current_q_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])  # Maximum Q-value for the next state
        self.q_table[state, action] = current_q_value + self.alpha * (reward + self.gamma * next_max - current_q_value)

"""## Boltzmann Q Learning Agent"""

class BoltzmannQLearningAgent:
    def __init__(self, num_states, num_actions, temperature=1.0, alpha=0.1, gamma=0.9):
        self.num_actions = num_actions
        self.temperature = temperature  # Exploration temperature
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = np.zeros((num_states, num_actions))

    def name(self):
        return 'Boltzmann Q-learning'

    def set_q_table(self, q_table):
        self.q_table = q_table

    def choose_action(self, state):
        # Calculate action probabilities using Boltzmann distribution
        action_values = self.q_table[state]
        exp_values = np.exp(action_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        chosen_action = np.random.choice(np.arange(self.num_actions), p=probabilities)
        return chosen_action

    def get_probs(self, state):
        # Return action probabilities for the given state
        action_values = self.q_table[state]
        exp_values = np.exp(action_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def update_q_table(self, state, action, reward, next_state):
        # Update Q-value based on action and reward
        current_q_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])  # Max Q-value for the next state
        # Standard Q-learning update rule
        self.q_table[state, action] = current_q_value + self.alpha * (reward + self.gamma * next_max - current_q_value)

"""## Lenient Boltzmann Q Learning Agent"""

class LenientBoltzmannQLearningAgent:
    def __init__(self, num_states, num_actions, temperature=1.0, alpha=0.1, gamma=0.9, leniency=10):
        self.num_actions = num_actions
        self.temperature = temperature  # Exploration temperature
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = np.zeros((num_states, num_actions))
        self.leniency = leniency  # Leniency threshold
        # Keep track of recent rewards for each action
        self.reward_memory = {action: [] for action in range(num_actions)}

    def name(self):
        return 'Lenient Boltzmann Q-learning'

    def set_q_table(self, q_table):
        self.q_table = q_table

    def choose_action(self, state):
        # Calculate action probabilities using Boltzmann distribution
        action_values = self.q_table[state]
        exp_values = np.exp(action_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        chosen_action = np.random.choice(np.arange(self.num_actions), p=probabilities)
        return chosen_action

    def get_probs(self, state):
        # Return action probabilities for the given state
        action_values = self.q_table[state]
        exp_values = np.exp(action_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def process_reward(self, action, reward):
        # Store the reward in the memory
        self.reward_memory[action].append(reward)

        # Only keep the most recent leniency rewards
        if len(self.reward_memory[action]) > self.leniency:
            self.reward_memory[action] = self.reward_memory[action][-self.leniency:]

        # Return the maximum of the stored rewards
        return max(self.reward_memory[action])

    def update_q_table(self, state, action, reward, next_state):
        # Process the reward with leniency
        lenient_reward = self.process_reward(action, reward)

        # Update Q-value based on lenient reward
        current_q_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])  # Max Q-value for the next state
        # Standard Q-learning update rule with lenient reward
        self.q_table[state, action] = current_q_value + self.alpha * (lenient_reward + self.gamma * next_max - current_q_value)

"""## Lenient Frequency Adjusted Q Learning Agent"""

class LenientFrequencyAdjustedQLearningAgent:
    def __init__(self, num_states, num_actions, leniency,
                 temperature, alpha, gamma, beta):
        self.num_states = num_states
        self.num_actions = num_actions
        self.leniency = leniency
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.q_table = np.zeros((num_states, num_actions))
        self.frequency_table = np.ones(num_actions)  # Start with frequency of 1 to avoid division by zero
        self.reward_samples = {action: [] for action in range(num_actions)}

    def name(self):
        return 'Lenient Frequency Adjusted Q-learning Agent'

    def set_q_table(self, q_table):
        self.q_table = q_table

    def choose_action(self, state):
        action_probabilities = np.exp(self.q_table[state] / self.temperature)
        action_probabilities /= np.sum(action_probabilities)
        action = np.random.choice(range(self.num_actions), p=action_probabilities)
        return action

    def get_probs(self, state):
        action_probabilities = np.exp(self.q_table[state] / self.temperature)
        action_probabilities /= np.sum(action_probabilities)
        return action_probabilities

    def process_reward(self, action, reward):
        self.reward_samples[action].append(reward)
        if len(self.reward_samples[action]) > self.leniency:
            recent_rewards = self.reward_samples[action][-self.leniency:]
        else:
            recent_rewards = self.reward_samples[action]
        max_reward = max(recent_rewards)
        return max_reward

    def update_q_table(self, state, action, reward, next_state):
        max_reward = self.process_reward(action, reward)
        self.frequency_table[action] += 1
        freq_adjustment = min(1, self.beta / self.frequency_table[action])
        adjusted_alpha = self.alpha * freq_adjustment
        best_future_q = np.max(self.q_table[next_state])  # Use Q-values of the next state
        self.q_table[state, action] += adjusted_alpha * (max_reward + self.gamma * best_future_q - self.q_table[state, action])

"""## Utility Functions"""

def play_episode(agent1, agent2, payoff_matrix, is3x3 = False):
    # Both agents choose actions (state is fixed or irrelevant)
    state = 0  # Placeholder state since the game is static
    agent1_action = agent1.choose_action(state)
    agent2_action = agent2.choose_action(state)

    # Retrieve combined rewards from the payoff matrix
    if (is3x3 == False):
        combined_rewards = payoff_matrix[agent1_action, agent2_action]
    else:
        reward_player1 = payoff_matrix[agent1_action, agent2_action]
        combined_rewards = reward_player1, -reward_player1

    # Unpack rewards for both players
    reward_player1, reward_player2 = combined_rewards

    # Update Q-tables (no next state needed in a static game)
    agent1.update_q_table(state, agent1_action, reward_player1, state)  # next_state is the same as state
    agent2.update_q_table(state, agent2_action, reward_player2, state)  # next_state is the same as state

    return reward_player1, reward_player2

def softmax(q_table):
    # Apply softmax function to each row of the Q-table
    exp_q = np.exp(q_table - np.max(q_table, axis=1, keepdims=True))  # For numerical stability
    return exp_q / np.sum(exp_q, axis=1, keepdims=True)

# Function to play multiple episodes and calculate average reward
def play_multiple_episodes(agent1, agent2, num_episodes, matrix_payoffs, is3x3):
    total_rewards = np.zeros(num_episodes)
    probs_0 = []
    probs_1 = []
    for episode in range(num_episodes):
        total_rewards[episode], _ = play_episode(agent1, agent2, matrix_payoffs, is3x3)
        prob_0 = softmax(agent1.q_table)
        prob_1 = softmax(agent2.q_table)
        probs_0.append(prob_0)
        probs_1.append(prob_1)
    return total_rewards, probs_0, probs_1

def generate_plot(matrix_payoffs, num_actions, game_name, is3x3):
    # Parameters for ε-greedy Q-learning
    epsilon = 0.15
    alpha = 0.001
    gamma = 0.9

    # Create instances of Epsilon-Greedy Q-learning Agent
    agent1 = EpsilonGreedyQLearningAgent(num_states=1, num_actions=num_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)
    agent2 = EpsilonGreedyQLearningAgent(num_states=1, num_actions=num_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)

    # Parameters for Boltzmann Q-learning
    temperature = 0.5
    alpha = 0.001
    gamma = 0.9

    # Create instances of Boltzmann Q Learning Agent
    boltzmann_agent1 = BoltzmannQLearningAgent(num_states=1, num_actions=num_actions, temperature=temperature, alpha=alpha, gamma=gamma)
    boltzmann_agent2 = BoltzmannQLearningAgent(num_states=1, num_actions=num_actions, temperature=temperature, alpha=alpha, gamma=gamma)

    # Parameters for Lenient Boltzmann Q-learning
    temperature = 1
    alpha = 0.001
    gamma = 0.9
    leniency = 10

    # Create instances of Lenient Boltzmann Q Learning Agent
    lenient_boltzmann_agent1 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions, temperature=temperature, alpha=alpha, gamma=gamma, leniency=leniency)
    lenient_boltzmann_agent2 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions, temperature=temperature, alpha=alpha, gamma=gamma, leniency=leniency)

    # Parameters for Lenient Frequency Adjusted Q-learning
    beta = 0.001
    temperature = 1
    alpha = 0.001
    gamma = 0.9
    leniency = 10

    # Create instances of Lenient Frequency Adjusted Q Learning Agent
    lfaq_agent1 = LenientFrequencyAdjustedQLearningAgent(num_states=1, num_actions=num_actions, leniency=leniency, temperature=temperature, alpha=alpha, gamma=gamma, beta=beta)
    lfaq_agent2 = LenientFrequencyAdjustedQLearningAgent(num_states=1, num_actions=num_actions, leniency=leniency, temperature=temperature, alpha=alpha, gamma=gamma, beta=beta)

    # Play multiple episodes and calculate average probabilities
    num_episodes = 10000

    def compute_average_probabilities(probs):
        cumulative_probs = np.cumsum(probs, axis=0)
        averages = cumulative_probs / (np.arange(1, len(probs) + 1)[:, np.newaxis])
        return averages[:, 0]  # Return the average probabilities for action 0

    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, matrix_payoffs, is3x3)
    avg_q_learning_probs_0_for_action_0 = compute_average_probabilities([prob[0] for prob in probs_0])
    avg_q_learning_probs_1_for_action_0 = compute_average_probabilities([prob[0] for prob in probs_1])

    rewards_boltzmann, probs_0, probs_1 = play_multiple_episodes(boltzmann_agent1, boltzmann_agent2, num_episodes, matrix_payoffs, is3x3)
    avg_boltzmann_probs_0_for_action_0 = compute_average_probabilities([prob[0] for prob in probs_0])
    avg_boltzmann_probs_1_for_action_0 = compute_average_probabilities([prob[0] for prob in probs_1])

    rewards_lenient_boltzmann, probs_0, probs_1 = play_multiple_episodes(lenient_boltzmann_agent1, lenient_boltzmann_agent2, num_episodes, matrix_payoffs, is3x3)
    avg_lenient_boltzmann_probs_0_for_action_0 = compute_average_probabilities([prob[0] for prob in probs_0])
    avg_lenient_boltzmann_probs_1_for_action_0 = compute_average_probabilities([prob[0] for prob in probs_1])

    rewards_lfaq, probs_0, probs_1 = play_multiple_episodes(lfaq_agent1, lfaq_agent2, num_episodes, matrix_payoffs, is3x3)
    avg_lfaq_probs_0_for_action_0 = compute_average_probabilities([prob[0] for prob in probs_0])
    avg_lfaq_probs_1_for_action_0 = compute_average_probabilities([prob[0] for prob in probs_1])

    # Plotting the average probabilities
    plt.figure(figsize=(10, 6))
    plt.plot(avg_q_learning_probs_0_for_action_0, label='ε-greedy Q-learning - Agent 1')
    plt.plot(avg_q_learning_probs_1_for_action_0, label='ε-greedy Q-learning - Agent 2')
    plt.plot(avg_boltzmann_probs_0_for_action_0, label='Boltzmann Q-learning - Agent 1')
    plt.plot(avg_boltzmann_probs_1_for_action_0, label='Boltzmann Q-learning - Agent 2')
    plt.plot(avg_lenient_boltzmann_probs_0_for_action_0, label='Lenient Boltzmann Q-learning - Agent 1')
    plt.plot(avg_lenient_boltzmann_probs_1_for_action_0, label='Lenient Boltzmann Q-learning - Agent 2')
    plt.plot(avg_lfaq_probs_0_for_action_0, label='Lenient Frequency Adjusted Q-learning - Agent 1')
    plt.plot(avg_lfaq_probs_1_for_action_0, label='Lenient Frequency Adjusted Q-learning - Agent 2')
    plt.title(f'Average Probability for Action 1: Trajectories for {game_name}')
    plt.xlabel('Episodes')
    plt.ylabel('Average Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

"""# Independent learning in benchmark matrix games

## Subsidy Game
"""

subsidy_game_payoffs = np.array([
    # Player 1 applies for subsidy, Player 2 applies for subsidy
    [[12, 12],  # Both players receive a payoff of 12. This scenario represents both players sharing the subsidy equally.

    # Player 1 applies for subsidy, Player 2 does not apply for subsidy
     [0, 11]],  # Player 1 receives a payoff of 0 (perhaps due to overutilization penalties or the subsidy being denied when not shared), while Player 2 receives a payoff of 11 (benefiting from not applying, possibly due to alternative rewards or avoiding penalties).

    # Player 1 does not apply for subsidy, Player 2 applies for subsidy
    [[11, 0],   # Player 1 receives a payoff of 11 (similarly benefiting as in the above case), while Player 2 receives a payoff of 0 (facing penalties or denial for the same reasons).

    # Player 1 does not apply for subsidy, Player 2 does not apply for subsidy
     [10, 10]]  # Both players receive a payoff of 10. This scenario represents both players not applying for the subsidy, possibly reflecting a baseline welfare level without the subsidy's influence.
])

player1_payoffs = np.array([
    [12, 0],  # Payoffs for player 1 when player 1 applies
    [11, 10]  # Payoffs for player 1 when player 1 does not apply
])

player2_payoffs = np.array([
    [12, 11],  # Payoffs for player 2 when player 2 applies
    [0, 10]    # Payoffs for player 2 when player 2 does not apply
])

subsidy_game = pyspiel.create_matrix_game(
    "subsidy", "Subsidy Game",
    ["Apply", "Not Apply"],  # Actions for player 1
    ["Apply", "Not Apply"],  # Actions for player 2
    player1_payoffs.tolist(),  # Player 1's payoffs
    player2_payoffs.tolist()   # Player 2's payoffs
)
generate_plot(subsidy_game_payoffs, subsidy_game.num_distinct_actions(), 'Subsidy Game', is3x3=False)

battle_of_sexes_payoffs = np.array([
    # Player 1 chooses Opera, Player 2 chooses Opera
    [[3, 2],  # Both players enjoy the event together, but Player 1 prefers Opera slightly more.
    # Player 1 chooses Opera, Player 2 chooses Football
     [0, 0]],  # Neither gets their preference, resulting in no payoff.
    # Player 1 chooses Football, Player 2 chooses Opera
    [[0, 0],   # Neither gets their preference, resulting in no payoff.
    # Player 1 chooses Football, Player 2 chooses Football
     [2, 3]]  # Both players enjoy the event together, but Player 2 prefers Football slightly more.
])

# Battle of the Sexes Payoffs
battle_of_sexes_payoffs_p1 = np.array([
    [3, 0],  # P1: Opera, P2: Opera | P1: Opera, P2: Football
    [0, 2]   # P1: Football, P2: Opera | P1: Football, P2: Football
])

battle_of_sexes_payoffs_p2 = np.array([
    [2, 0],  # P1: Opera, P2: Opera | P1: Opera, P2: Football
    [0, 3]   # P1: Football, P2: Opera | P1: Football, P2: Football
])

battle_of_sexes_game = pyspiel.create_matrix_game(
    "battle_of_sexes", "Battle of the Sexes",
    ["Opera", "Football"],  # Actions for player 1
    ["Opera", "Football"],  # Actions for player 2
    battle_of_sexes_payoffs_p1.tolist(),  # Player 1's payoffs
    battle_of_sexes_payoffs_p2.tolist()   # Player 2's payoffs
)
generate_plot(battle_of_sexes_payoffs, battle_of_sexes_game.num_distinct_actions(), 'Battle of the Sexes', is3x3=False)

prisoners_dilemma_payoffs = np.array([
    # Both players Confess
    [[-1, -1],  # Mutual cooperation leads to a modest penalty for both.
    # Player 1 Confess, Player 2 Deceive
     [-4, 0]],  # Player 1 suffers a significant penalty, while Player 2 goes free.
    # Player 1 Deceive, Player 2 Confess
    [[0, -4],   # Player 2 suffers a significant penalty, while Player 1 goes free.
    # Both players Deceive
     [-3, -3]]  # Mutual deceive leads to a severe penalty for both.
])

prisoners_dilemma_payoffs_p1 = np.array([
    [-1, -4],  # P1: Confess, P2: Confess | P1: Confess, P2: Deceive
    [0, -3]    # P1: Deceive, P2: Confess | P1: Deceive, P2: Deceive
])

prisoners_dilemma_payoffs_p2 = np.array([
    [-1, 0],   # P1: Confess, P2: Confess | P1: Confess, P2: Deceive
    [-4, -3]   # P1: Deceive, P2: Confess | P1: Deceive, P2: Deceive
])

prisoners_dilemma_game = pyspiel.create_matrix_game(
    "prisoners_dilemma", "Prisoner's Dilemma",
    ["Cooperate", "Defect"],  # Actions for player 1
    ["Cooperate", "Defect"],  # Actions for player 2
    prisoners_dilemma_payoffs_p1.tolist(),  # Player 1's payoffs
    prisoners_dilemma_payoffs_p2.tolist()   # Player 2's payoffs
)
generate_plot(prisoners_dilemma_payoffs, prisoners_dilemma_game.num_distinct_actions(), 'Prisoners Dilemma', is3x3=False)

rock_paper_scissors_payoffs = np.array([[0,-0.05,0.25],
                                        [0.05,0,-0.5],
                                        [-0.25,0.5,0]])

# If Player 1 chooses Rock (row 1):

# If Player 2 chooses Rock (column 1), both players receive a payoff of 0.
# If Player 2 chooses Paper (column 2), Player 1 receives a penalty of -0.05, and Player 2 receives a reward of 0.05.
# If Player 2 chooses Scissors (column 3), Player 1 receives a reward of 0.25, and Player 2 receives a penalty of -0.25.


# If Player 1 chooses Paper (row 2):

# If Player 2 chooses Rock (column 1), Player 1 receives a reward of 0.05, and Player 2 receives a penalty of -0.05.
# If Player 2 chooses Paper (column 2), both players receive a payoff of 0.
# If Player 2 chooses Scissors (column 3), Player 1 receives a penalty of -0.5, and Player 2 receives a reward of 0.5.


# If Player 1 chooses Scissors (row 3):

# If Player 2 chooses Rock (column 1), Player 1 receives a penalty of -0.25, and Player 2 receives a reward of 0.25.
# If Player 2 chooses Paper (column 2), Player 1 receives a reward of 0.5, and Player 2 receives a penalty of -0.5.
# If Player 2 chooses Scissors (column 3), both players receive a payoff of 0.

rock_paper_scissors = pyspiel.create_matrix_game("rock_paper_scissors",
                           "Biased Rock-Paper-Scissors",
                           ["R", "P", "S"], ["R", "P", "S"],
                           rock_paper_scissors_payoffs,
                           rock_paper_scissors_payoffs.T)

generate_plot(rock_paper_scissors_payoffs, rock_paper_scissors.num_distinct_actions(), 'Rock Paper Scissors', is3x3 = True)

"""# Dynamics of learning in benchmark matrix games:

## Subsidy Game

Epsilon Greedy Q Learning
"""

payoff_matrix = subsidy_game_payoffs

# Define the replicator dynamics
def replicator_dynamics(p1, p2):
    # Expected payoffs for Player 1
    pi1_a1 = p2 * payoff_matrix[0, 0, 0] + (1 - p2) * payoff_matrix[0, 1, 0]
    pi1_a2 = p2 * payoff_matrix[1, 0, 0] + (1 - p2) * payoff_matrix[1, 1, 0]
    avg_pi1 = p1 * pi1_a1 + (1 - p1) * pi1_a2

    # Expected payoffs for Player 2
    pi2_a1 = p1 * payoff_matrix[0, 0, 1] + (1 - p1) * payoff_matrix[1, 0, 1]
    pi2_a2 = p1 * payoff_matrix[0, 1, 1] + (1 - p1) * payoff_matrix[1, 1, 1]
    avg_pi2 = p2 * pi2_a1 + (1 - p2) * pi2_a2

    # Replicator equations
    dp1 = p1 * (pi1_a1 - avg_pi1)
    dp2 = p2 * (pi2_a1 - avg_pi2)

    return dp1, dp2

# Grid of probabilities
p1_vals = np.linspace(0, 1, 20)
p2_vals = np.linspace(0, 1, 20)
P1, P2 = np.meshgrid(p1_vals, p2_vals)

# Compute the derivatives
dP1, dP2 = np.zeros(P1.shape), np.zeros(P2.shape)
for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        dp1, dp2 = replicator_dynamics(P1[i, j], P2[i, j])
        dP1[i, j] = dp1
        dP2[i, j] = dp2

# Plot the phase plot
plt.figure(figsize=(8, 8))
plt.quiver(P1, P2, dP1/20.0, dP2/20.0, angles='xy', scale_units='xy', scale=1, color='black', alpha = 1)
plt.xlabel('Probability of Player 1 choosing Action 1')
plt.ylabel('Probability of Player 2 choosing Action 1')
plt.title('Replicator Dynamics for Subsidy Game - Epsilon-Greedy Q-Learning')
plt.grid(True)
plt.show()

# Parameters
num_states = 1
num_actions = 2

# Generate random start points for learning trajectories
start_points = [np.random.rand(num_states, num_actions) - 0.5 for _ in range(10)]
colors = plt.cm.get_cmap('hsv', len(start_points))
colors = [colors(i) for i in range(len(start_points))]
colors_i = 0

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 8))

 
# The quiver plot should show the replicator dynamics vector field
ax.quiver(P1, P2, dP1/20.0, dP2/20.0, angles='xy', scale_units='xy', scale=1, color='black', alpha=0.25)

# Labeling
ax.set_xlabel('Probability of Player 1 choosing Action 1')
ax.set_ylabel('Probability of Player 2 choosing Action 1')
ax.set_title('Replicator Dynamics for Subsidy Game - Epsilon-Greedy Q-Learning')
ax.grid(True)

# Simulate learning trajectories and plot them
for q_table in start_points:
    epsilon = 0.15
    alpha = 0.001
    gamma = 0.9

    # Create instances of RLAgent for both players
    agent1 = EpsilonGreedyQLearningAgent(num_states=num_states, num_actions=num_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)
    agent2 = EpsilonGreedyQLearningAgent(num_states=num_states, num_actions=num_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 1000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    ax.scatter(q_learning_probs_0_for_action_0[0], q_learning_probs_1_for_action_0[0], s=25, color = colors[colors_i])

    # Plot the trajectory on the same axes as the quiver plot
    ax.plot(q_learning_probs_0_for_action_0, q_learning_probs_1_for_action_0, linestyle='-', color=colors[colors_i], alpha=1.0)
    colors_i += 1

# Display the plot
plt.show()

"""Boltzmann Q Learning"""

import numpy as np
import matplotlib.pyplot as plt

payoff_matrix = subsidy_game_payoffs

temperature = 10

# Define the Lenient Boltzmann Q-learning adjusted replicator dynamics
def replicator_dynamics(p1, p2, temperature):
    # Expected payoffs for Player 1
    pi1_a1 = p2 * payoff_matrix[0, 0, 0] + (1 - p2) * payoff_matrix[0, 1, 0]
    pi1_a2 = p2 * payoff_matrix[1, 0, 0] + (1 - p2) * payoff_matrix[1, 1, 0]
    avg_pi1 = p1 * pi1_a1 + (1 - p1) * pi1_a2

    # Expected payoffs for Player 2
    pi2_a1 = p1 * payoff_matrix[0, 0, 1] + (1 - p1) * payoff_matrix[1, 0, 1]
    pi2_a2 = p1 * payoff_matrix[0, 1, 1] + (1 - p1) * payoff_matrix[1, 1, 1]
    avg_pi2 = p2 * pi2_a1 + (1 - p2) * pi2_a2

    # Replicator equations
    dp1 = p1 * (pi1_a1 - avg_pi1)
    dp2 = p2 * (pi2_a1 - avg_pi2)

    return dp1, dp2

# Grid of probabilities
p1_vals = np.linspace(0, 1, 20)
p2_vals = np.linspace(0, 1, 20)
P1, P2 = np.meshgrid(p1_vals, p2_vals)

# Compute the derivatives
dP1, dP2 = np.zeros(P1.shape), np.zeros(P2.shape)
for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        dp1, dp2 = replicator_dynamics(P1[i, j], P2[i, j], temperature)
        dP1[i, j] = dp1
        dP2[i, j] = dp2

# Plot the phase plot
plt.figure(figsize=(8, 8))
plt.quiver(P1, P2, dP1/20.0, dP2/20.0, angles='xy', scale_units='xy', scale=1, color='black')
plt.xlabel('Probability of Player 1 choosing Action 1')
plt.ylabel('Probability of Player 2 choosing Action 1')
plt.title('Replicator Dynamics for Subsidy Game - for Boltzmann Q-Learning')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_states = 1
num_actions = 2

# Generate random start points for learning trajectories
start_points = [np.random.rand(num_states, num_actions) - 0.5 for _ in range(10)]
colors = plt.cm.get_cmap('hsv', len(start_points))
colors = [colors(i) for i in range(len(start_points))]
colors_i = 0

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 8))

 
# The quiver plot should show the replicator dynamics vector field
ax.quiver(P1, P2, dP1/20.0, dP2/20.0, angles='xy', scale_units='xy', scale=1, color='black', alpha=0.25)

# Labeling
ax.set_xlabel('Probability of Player 1 choosing Action 1')
ax.set_ylabel('Probability of Player 2 choosing Action 1')
ax.set_title('Replicator Dynamics for Subsidy Game - Boltzmann Q-Learning')
ax.grid(True)

# Simulate learning trajectories and plot them
for q_table in start_points:
    temperature = 0.5
    alpha = 0.001
    gamma = 0.9

    # Create instances of Boltzmann Q Learning Agent
    agent1 = BoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                              temperature=temperature, alpha=alpha, gamma=gamma)
    agent2 = BoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                            temperature=temperature, alpha=alpha, gamma=gamma)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 1000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    ax.scatter(q_learning_probs_0_for_action_0[0], q_learning_probs_1_for_action_0[0], s=25, color = colors[colors_i])


    # Plot the trajectory on the same axes as the quiver plot
    ax.plot(q_learning_probs_0_for_action_0, q_learning_probs_1_for_action_0, linestyle='-', color=colors[colors_i], alpha=1.0)
    colors_i += 1

# Display the plot
plt.show()

"""Lenient Boltzmann Q Learning"""

!unzip /content/new-ml-project-kul-master.zip

# Commented out IPython magic to ensure Python compatibility.
# %cd new-ml-project-kul-master/

from Assets.Game import Game
from Assets.Graph import Graph

legal_games = ["rock_paper_scissors", "subsidy_game", "Battle_of_the_sexes", "prisoners_dilemma"]
game_name = "subsidy_game"
alpha = 0.001
tau = 0.5
game = Game(game_name)
kappa = 10

fig = Graph.compute_vector_field(game, alpha, tau , kappa=kappa, normalise= True)

# title = f"{game.get_game_name()}\n"
# title += f"| alpha={alpha}, tau={tau}, kappa={kappa}"

fig.update_layout(xaxis_title='Probability of Player 1 picking Action 1',
                yaxis_title='Probability of Player 2 picking Action 1',
                # legend_title='Traces',
                xaxis=dict(range=[0, 1], scaleanchor="y"),  # Set X axis range and link scale to Y axis
                yaxis=dict(range=[0, 1], scaleratio=1),     # Set Y axis range and ensure square aspect ratio
                width=600,  # Set the width of the plot
                height=600)  # Set the height of the plot to the same value as width to make it square

# Show the plot
fig.show()

# @title
colors = [
    'red',          # 1
    'blue',         # 2
    'green',        # 3
    'purple',       # 4
    'orange',       # 5
    'cyan',         # 6
    'magenta',      # 7
    'yellow',       # 8
    'black',        # 9
    'brown',        # 10
    'pink',         # 11
    'lime',         # 12
    'indigo',       # 13
    'violet',       # 14
    'turquoise',    # 15
    'gold',         # 16
    'darkgreen',    # 17
    'darkred',      # 18
    'navy',         # 19
    'coral',        # 20
    'teal',         # 21
    'olive',        # 22
    'maroon',       # 23
    'salmon',       # 24
    'plum'          # 25
]

import plotly.graph_objects as go

# Parameters
num_states = 1
num_actions = 2

# Generate random start points for learning trajectories
start_points = [2 * np.random.rand(num_states, num_actions) - 1 for _ in range(10)]

colors_i = 0

for q_table in start_points:
    temperature = 0.5
    alpha = 0.0003
    gamma = 0.9
    leniency = 10

    # Create instances of Lenient Boltzmann Q Learning Agent
    agent1 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                            temperature=temperature, alpha=alpha, gamma=gamma, leniency=leniency)
    agent2 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                            temperature=temperature, alpha=alpha, gamma=gamma, leniency=leniency)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 10000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    # Add initial scatter point
    fig.add_trace(go.Scatter(
        x=[q_learning_probs_0_for_action_0[0]],
        y=[q_learning_probs_1_for_action_0[0]],
        mode='markers',
        marker=dict(size=8, color=colors[colors_i]),
        # name=f"Start {colors_i+1}"
    ))

    # Plot the trajectory on the same figure
    fig.add_trace(go.Scatter(
        x=q_learning_probs_0_for_action_0,
        y=q_learning_probs_1_for_action_0,
        mode='lines',
        line=dict(color=colors[colors_i], width=2),
        # name=f"Trajectory {colors_i+1}"
    ))

    colors_i += 1

fig.show()

"""Lenient Frequency Adjusted Q Learning"""

import numpy as np
import matplotlib.pyplot as plt

# Payoff matrix for Player 1 and Player 2
payoff_matrix = subsidy_game_payoffs

# Frequency Adjusted Boltzmann parameters
temperature = 10  # Temperature parameter for Boltzmann exploration
alpha = 0.1        # Base learning rate
beta = 0.5       # Scaling parameter for frequency adjustment

def replicator_dynamics_freq_adj(p1, p2, temperature, alpha, beta):
    # Boltzmann probabilities for Player 1 and Player 2
    exp_q1 = np.exp(payoff_matrix[:, :, 0] / temperature)
    exp_q2 = np.exp(payoff_matrix[:, :, 1] / temperature)

    boltzmann_p1 = exp_q1 / np.sum(exp_q1, axis=0)
    boltzmann_p2 = exp_q2 / np.sum(exp_q2, axis=1)

    # Expected payoffs for Player 1
    pi1_a1 = p2 * payoff_matrix[0, 0, 0] + (1 - p2) * payoff_matrix[0, 1, 0]
    pi1_a2 = p2 * payoff_matrix[1, 0, 0] + (1 - p2) * payoff_matrix[1, 1, 0]
    avg_pi1 = p1 * pi1_a1 + (1 - p1) * pi1_a2

    # Expected payoffs for Player 2
    pi2_a1 = p1 * payoff_matrix[0, 0, 1] + (1 - p1) * payoff_matrix[1, 0, 1]
    pi2_a2 = p1 * payoff_matrix[0, 1, 1] + (1 - p1) * payoff_matrix[1, 1, 1]
    avg_pi2 = p2 * pi2_a1 + (1 - p2) * pi2_a2

    # Replicator equations with frequency adjustment
    freq_p1_a1 = np.sum(boltzmann_p1[:, 0])  # Frequency of choosing action 1
    freq_p1_a2 = np.sum(boltzmann_p1[:, 1])  # Frequency of choosing action 2
    freq_p2_a1 = np.sum(boltzmann_p2[0, :])  # Frequency of choosing action 1
    freq_p2_a2 = np.sum(boltzmann_p2[1, :])  # Frequency of choosing action 2

    # Adjust learning rates based on frequencies
    freq_adjustment_p1_a1 = min(1, beta / freq_p1_a1)
    freq_adjustment_p1_a2 = min(1, beta / freq_p1_a2)
    freq_adjustment_p2_a1 = min(1, beta / freq_p2_a1)
    freq_adjustment_p2_a2 = min(1, beta / freq_p2_a2)

    dp1 = p1 * (pi1_a1 - avg_pi1) * freq_adjustment_p1_a1
    dp2 = p2 * (pi2_a1 - avg_pi2) * freq_adjustment_p2_a1

    return dp1, dp2

# Grid of probabilities
p1_vals = np.linspace(0, 1, 20)
p2_vals = np.linspace(0, 1, 20)
P1, P2 = np.meshgrid(p1_vals, p2_vals)

# Compute the derivatives
dP1, dP2 = np.zeros(P1.shape), np.zeros(P2.shape)
for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        dp1, dp2 = replicator_dynamics_freq_adj(P1[i, j], P2[i, j], temperature, alpha, beta)
        dP1[i, j] = dp1
        dP2[i, j] = dp2

# Plot the phase plot
plt.figure(figsize=(8, 8))
plt.quiver(P1, P2, dP1/20.0, dP2/20.0, angles='xy', scale_units='xy', scale=1, color='black')
plt.xlabel('Probability of Player 1 choosing Action 1')
plt.ylabel('Probability of Player 2 choosing Action 1')
plt.title('Replicator Dynamics for Subsidy Game - Lenient Frequency Boltzmann Q-Learning')
plt.grid(True)
plt.show()

import numpy as np
import plotly.graph_objects as go
from Assets.Game import Game
from Assets.Graph import Graph

# Define the game and parameters
legal_games = ["rock_paper_scissors", "subsidy_game", "Battle_of_the_sexes", "prisoners_dilemma"]
game_name = "subsidy_game"
alpha = 0.01
tau = 0.5
game = Game(game_name)
kappa = 1

# Compute the vector field
fig = Graph.compute_vector_field(game, alpha, tau, kappa=kappa, normalise=True)

# Layout updates for the vector field plot
fig.update_layout(
    xaxis_title='Probability of Player 1 picking Action 1',
    yaxis_title='Probability of Player 2 picking Action 1',
    xaxis=dict(range=[0, 1], scaleanchor="y"),  # Set X axis range and link scale to Y axis
    yaxis=dict(range=[0, 1], scaleratio=1),     # Set Y axis range and ensure square aspect ratio
    width=600,  # Set the width of the plot
    height=600  # Set the height of the plot to the same value as width to make it square
)

# Generate random start points for learning trajectories
num_states = 1
num_actions = 2
start_points = [2 * np.random.rand(num_states, num_actions) - 1 for _ in range(10)]
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'brown']
colors_i = 0

payoff_matrix = subsidy_game_payoffs

# Simulate learning trajectories and plot them
for q_table in start_points[:5]:
    # Parameters for Lenient Frequency Adjusted Q-learning
    beta = 3
    temperature = 0.5
    alpha = 0.002
    gamma = 0.9
    leniency = 5

    # Create instances of Lenient Frequency Adjusted Q Learning Agent
    agent1 = LenientFrequencyAdjustedQLearningAgent(num_states=1, num_actions=num_actions, leniency=leniency, temperature=temperature, alpha=alpha, gamma=gamma, beta=beta)
    agent2 = LenientFrequencyAdjustedQLearningAgent(num_states=1, num_actions=num_actions, leniency=leniency, temperature=temperature, alpha=alpha, gamma=gamma, beta=beta)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 1000000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    # Add initial scatter point
    fig.add_trace(go.Scatter(
        x=[q_learning_probs_0_for_action_0[0]],
        y=[q_learning_probs_1_for_action_0[0]],
        mode='markers',
        marker=dict(size=8, color=colors[colors_i]),
        name=f"Start {colors_i+1}"
    ))

    # Plot the trajectory on the same figure
    fig.add_trace(go.Scatter(
        x=q_learning_probs_0_for_action_0[::100],
        y=q_learning_probs_1_for_action_0[::100],
        mode='lines',
        line=dict(color=colors[colors_i], width=2),
        name=f"Trajectory {colors_i+1}"
    ))

    colors_i += 1

# Show the plot with learning trajectories overlaying the vector field
fig.show()

"""## Battle of the Sexes

Epsilon Greedy Q Learning
"""

payoff_matrix = battle_of_sexes_payoffs

# Define the replicator dynamics
def replicator_dynamics(p1, p2):
    # Expected payoffs for Player 1
    pi1_a1 = p2 * payoff_matrix[0, 0, 0] + (1 - p2) * payoff_matrix[0, 1, 0]
    pi1_a2 = p2 * payoff_matrix[1, 0, 0] + (1 - p2) * payoff_matrix[1, 1, 0]
    avg_pi1 = p1 * pi1_a1 + (1 - p1) * pi1_a2

    # Expected payoffs for Player 2
    pi2_a1 = p1 * payoff_matrix[0, 0, 1] + (1 - p1) * payoff_matrix[1, 0, 1]
    pi2_a2 = p1 * payoff_matrix[0, 1, 1] + (1 - p1) * payoff_matrix[1, 1, 1]
    avg_pi2 = p2 * pi2_a1 + (1 - p2) * pi2_a2

    # Replicator equations
    dp1 = p1 * (pi1_a1 - avg_pi1)
    dp2 = p2 * (pi2_a1 - avg_pi2)

    return dp1, dp2

# Grid of probabilities
p1_vals = np.linspace(0, 1, 20)
p2_vals = np.linspace(0, 1, 20)
P1, P2 = np.meshgrid(p1_vals, p2_vals)

# Compute the derivatives
dP1, dP2 = np.zeros(P1.shape), np.zeros(P2.shape)
for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        dp1, dp2 = replicator_dynamics(P1[i, j], P2[i, j])
        dP1[i, j] = dp1
        dP2[i, j] = dp2

# Plot the phase plot
plt.figure(figsize=(8, 8))
plt.quiver(P1, P2, dP1/10.0, dP2/10.0, angles='xy', scale_units='xy', scale=1, color='black', alpha = 1)
plt.xlabel('Probability of Player 1 choosing Action 1')
plt.ylabel('Probability of Player 2 choosing Action 1')
plt.title('Replicator Dynamics for Battle of the Sexes - Epsilon-Greedy Q-Learning')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_states = 1
num_actions = 2

# Generate random start points for learning trajectories
start_points = [np.random.rand(num_states, num_actions) - 0.5 for _ in range(10)]
colors = plt.cm.get_cmap('hsv', len(start_points))
colors = [colors(i) for i in range(len(start_points))]
colors_i = 0

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 8))

 
# The quiver plot should show the replicator dynamics vector field
ax.quiver(P1, P2, dP1/10.0, dP2/10.0, angles='xy', scale_units='xy', scale=1, color='black', alpha=0.25)

# Labeling
ax.set_xlabel('Probability of Player 1 choosing Action 1')
ax.set_ylabel('Probability of Player 2 choosing Action 1')
ax.set_title('Replicator Dynamics for Battle of the Sexes - Epsilon-Greedy Q-Learning')
ax.grid(True)

# Simulate learning trajectories and plot them
for q_table in start_points:
    epsilon = 0.001
    alpha = 0.001
    gamma = 0.9

    # Create instances of RLAgent for both players
    agent1 = EpsilonGreedyQLearningAgent(num_states=num_states, num_actions=num_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)
    agent2 = EpsilonGreedyQLearningAgent(num_states=num_states, num_actions=num_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 10000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    ax.scatter(q_learning_probs_0_for_action_0[0], q_learning_probs_1_for_action_0[0], s=25, color = colors[colors_i])

    # Plot the trajectory on the same axes as the quiver plot
    ax.plot(q_learning_probs_0_for_action_0, q_learning_probs_1_for_action_0, linestyle='-', color=colors[colors_i], alpha=1.0)
    colors_i += 1

# Display the plot
plt.show()

"""Boltzmann Q Learning"""

payoff_matrix = battle_of_sexes_payoffs

temperature = 0.1

# Define the Lenient Boltzmann Q-learning adjusted replicator dynamics
def replicator_dynamics(p1, p2, temperature):
    # Expected payoffs for Player 1
    pi1_a1 = p2 * payoff_matrix[0, 0, 0] + (1 - p2) * payoff_matrix[0, 1, 0]
    pi1_a2 = p2 * payoff_matrix[1, 0, 0] + (1 - p2) * payoff_matrix[1, 1, 0]
    avg_pi1 = p1 * pi1_a1 + (1 - p1) * pi1_a2

    # Expected payoffs for Player 2
    pi2_a1 = p1 * payoff_matrix[0, 0, 1] + (1 - p1) * payoff_matrix[1, 0, 1]
    pi2_a2 = p1 * payoff_matrix[0, 1, 1] + (1 - p1) * payoff_matrix[1, 1, 1]
    avg_pi2 = p2 * pi2_a1 + (1 - p2) * pi2_a2

    # Replicator equations
    dp1 = p1 * (pi1_a1 - avg_pi1)
    dp2 = p2 * (pi2_a1 - avg_pi2)

    return dp1, dp2

# Grid of probabilities
p1_vals = np.linspace(0, 1, 20)
p2_vals = np.linspace(0, 1, 20)
P1, P2 = np.meshgrid(p1_vals, p2_vals)

# Compute the derivatives
dP1, dP2 = np.zeros(P1.shape), np.zeros(P2.shape)
for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        dp1, dp2 = replicator_dynamics(P1[i, j], P2[i, j], temperature)
        dP1[i, j] = dp1
        dP2[i, j] = dp2

# Plot the phase plot
plt.figure(figsize=(8, 8))
plt.quiver(P1, P2, dP1/10.0, dP2/10.0, angles='xy', scale_units='xy', scale=1, color='black')
plt.xlabel('Probability of Player 1 choosing Action 1')
plt.ylabel('Probability of Player 2 choosing Action 1')
plt.title('Replicator Dynamics for Battle of the Sexes - for Boltzmann Q-Learning')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_states = 1
num_actions = 2

# Generate random start points for learning trajectories
start_points = [np.random.rand(num_states, num_actions) - 0.5 for _ in range(10)]
colors = plt.cm.get_cmap('hsv', len(start_points))
colors = [colors(i) for i in range(len(start_points))]
colors_i = 0

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 8))

 
# The quiver plot should show the replicator dynamics vector field
ax.quiver(P1, P2, dP1/10.0, dP2/10.0, angles='xy', scale_units='xy', scale=1, color='black', alpha=0.25)

# Labeling
ax.set_xlabel('Probability of Player 1 choosing Action 1')
ax.set_ylabel('Probability of Player 2 choosing Action 1')
ax.set_title('Replicator Dynamics for Battle of the Sexes - Boltzmann Q-Learning')
ax.grid(True)

# Simulate learning trajectories and plot them
for q_table in start_points:
    temperature = 0.05
    alpha = 0.001
    gamma = 0.9

    # Create instances of Boltzmann Q Learning Agent
    agent1 = BoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                              temperature=temperature, alpha=alpha, gamma=gamma)
    agent2 = BoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                            temperature=temperature, alpha=alpha, gamma=gamma)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 10000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    ax.scatter(q_learning_probs_0_for_action_0[0], q_learning_probs_1_for_action_0[0], s=25, color = colors[colors_i])

    # Plot the trajectory on the same axes as the quiver plot
    ax.plot(q_learning_probs_0_for_action_0, q_learning_probs_1_for_action_0, linestyle='-', color=colors[colors_i], alpha=1.0)
    colors_i += 1

# Display the plot
plt.show()

"""Lenient Boltzmann Q Learning"""

from Assets.Game import Game
from Assets.Graph import Graph

legal_games = ["rock_paper_scissors", "subsidy_game", "Battle_of_the_sexes", "prisoners_dilemma"]
game_name = "Battle_of_the_sexes"
alpha = 0.001
tau = 0.5
game = Game(game_name)
kappa = 5

fig = Graph.compute_vector_field(game, alpha, tau , kappa=kappa, normalise= True)

# title = f"{game.get_game_name()}\n"
# title += f"| alpha={alpha}, tau={tau}, kappa={kappa}"

fig.update_layout(xaxis_title='Probability of Player 1 picking Action 1',
                yaxis_title='Probability of Player 2 picking Action 1',
                # legend_title='Traces',
                xaxis=dict(range=[0, 1], scaleanchor="y"),  # Set X axis range and link scale to Y axis
                yaxis=dict(range=[0, 1], scaleratio=1),     # Set Y axis range and ensure square aspect ratio
                width=600,  # Set the width of the plot
                height=600)  # Set the height of the plot to the same value as width to make it square

# Show the plot
fig.show()

import plotly.graph_objects as go

# Parameters
num_states = 1
num_actions = 2

# Generate random start points for learning trajectories
start_points = [1 * np.random.rand(num_states, num_actions) - 0.5 for _ in range(10)]

colors_i = 0

payoff_matrix = battle_of_sexes_payoffs

for q_table in start_points:
    temperature = 0.5
    alpha = 0.001
    gamma = 0.9
    leniency = 5

    # Create instances of Lenient Boltzmann Q Learning Agent
    agent1 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                            temperature=temperature, alpha=alpha, gamma=gamma, leniency=leniency)
    agent2 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                            temperature=temperature, alpha=alpha, gamma=gamma, leniency=leniency)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 10000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    # Add initial scatter point
    fig.add_trace(go.Scatter(
        x=[q_learning_probs_0_for_action_0[0]],
        y=[q_learning_probs_1_for_action_0[0]],
        mode='markers',
        marker=dict(size=8, color=colors[colors_i]),
        # name=f"Start {colors_i+1}"
    ))

    # Plot the trajectory on the same figure
    fig.add_trace(go.Scatter(
        x=q_learning_probs_0_for_action_0,
        y=q_learning_probs_1_for_action_0,
        mode='lines',
        line=dict(color=colors[colors_i], width=2),
        # name=f"Trajectory {colors_i+1}"
    ))

    colors_i += 1

fig.show()

"""Lenient Frequency Adjusted Q Learning"""

import numpy as np
import plotly.graph_objects as go
from Assets.Game import Game
from Assets.Graph import Graph

# Define the game and parameters
legal_games = ["rock_paper_scissors", "subsidy_game", "Battle_of_the_sexes", "prisoners_dilemma"]
game_name = "Battle_of_the_sexes"
alpha = 0.001
tau = 0.5
game = Game(game_name)
kappa = 5

# Compute the vector field
fig = Graph.compute_vector_field(game, alpha, tau, kappa=kappa, normalise=True)

# Layout updates for the vector field plot
fig.update_layout(
    xaxis_title='Probability of Player 1 picking Action 1',
    yaxis_title='Probability of Player 2 picking Action 1',
    xaxis=dict(range=[0, 1], scaleanchor="y"),  # Set X axis range and link scale to Y axis
    yaxis=dict(range=[0, 1], scaleratio=1),     # Set Y axis range and ensure square aspect ratio
    width=600,  # Set the width of the plot
    height=600  # Set the height of the plot to the same value as width to make it square
)

# Generate random start points for learning trajectories
num_states = 1
num_actions = 2
start_points = [2 * np.random.rand(num_states, num_actions) - 1 for _ in range(10)]
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'brown']
colors_i = 0

payoff_matrix = battle_of_sexes_payoffs

# Simulate learning trajectories and plot them
for q_table in start_points:
    # Parameters for Lenient Frequency Adjusted Q-learning
    beta = 0.75
    temperature = 0.01
    alpha = 0.05
    gamma = 0.9
    leniency = 5

    # Create instances of Lenient Frequency Adjusted Q Learning Agent
    agent1 = LenientFrequencyAdjustedQLearningAgent(num_states=1, num_actions=num_actions, leniency=leniency, temperature=temperature, alpha=alpha, gamma=gamma, beta=beta)
    agent2 = LenientFrequencyAdjustedQLearningAgent(num_states=1, num_actions=num_actions, leniency=leniency, temperature=temperature, alpha=alpha, gamma=gamma, beta=beta)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 10000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    # Add initial scatter point
    fig.add_trace(go.Scatter(
        x=[q_learning_probs_0_for_action_0[0]],
        y=[q_learning_probs_1_for_action_0[0]],
        mode='markers',
        marker=dict(size=8, color=colors[colors_i]),
        name=f"Start {colors_i+1}"
    ))

    # Plot the trajectory on the same figure
    fig.add_trace(go.Scatter(
        x=q_learning_probs_0_for_action_0,
        y=q_learning_probs_1_for_action_0,
        mode='lines',
        line=dict(color=colors[colors_i], width=2),
        name=f"Trajectory {colors_i+1}"
    ))

    colors_i += 1

# Show the plot with learning trajectories overlaying the vector field
fig.show()

"""### Prisoner's Dilemma

Epsilon Greedy Q Learning
"""

prisoners_dilemma_payoffs + 4

payoff_matrix = prisoners_dilemma_payoffs + 4

# Define the replicator dynamics
def replicator_dynamics(p1, p2):
    # Expected payoffs for Player 1
    pi1_a1 = p2 * payoff_matrix[0, 0, 0] + (1 - p2) * payoff_matrix[0, 1, 0]
    pi1_a2 = p2 * payoff_matrix[1, 0, 0] + (1 - p2) * payoff_matrix[1, 1, 0]
    avg_pi1 = p1 * pi1_a1 + (1 - p1) * pi1_a2

    # Expected payoffs for Player 2
    pi2_a1 = p1 * payoff_matrix[0, 0, 1] + (1 - p1) * payoff_matrix[1, 0, 1]
    pi2_a2 = p1 * payoff_matrix[0, 1, 1] + (1 - p1) * payoff_matrix[1, 1, 1]
    avg_pi2 = p2 * pi2_a1 + (1 - p2) * pi2_a2

    # Replicator equations
    dp1 = p1 * (pi1_a1 - avg_pi1)
    dp2 = p2 * (pi2_a1 - avg_pi2)

    return dp1, dp2

# Grid of probabilities
p1_vals = np.linspace(0, 1, 20)
p2_vals = np.linspace(0, 1, 20)
P1, P2 = np.meshgrid(p1_vals, p2_vals)

# Compute the derivatives
dP1, dP2 = np.zeros(P1.shape), np.zeros(P2.shape)
for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        dp1, dp2 = replicator_dynamics(P1[i, j], P2[i, j])
        dP1[i, j] = dp1
        dP2[i, j] = dp2

# Plot the phase plot
plt.figure(figsize=(8, 8))
plt.quiver(P1, P2, dP1/10.0, dP2/10.0, angles='xy', scale_units='xy', scale=1, color='black', alpha = 1)
plt.xlabel('Probability of Player 1 choosing Action 1')
plt.ylabel('Probability of Player 2 choosing Action 1')
plt.title('Replicator Dynamics for Prisoners Dilemma - Epsilon-Greedy Q-Learning')
plt.grid(True)
plt.show()

# Parameters
num_states = 1
num_actions = 2

# Generate random start points for learning trajectories
start_points = [np.random.rand(num_states, num_actions) - 0.5 for _ in range(10)]
colors = plt.cm.get_cmap('hsv', len(start_points))
colors = [colors(i) for i in range(len(start_points))]
colors_i = 0

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 8))

 
# The quiver plot should show the replicator dynamics vector field
ax.quiver(P1, P2, dP1/10.0, dP2/10.0, angles='xy', scale_units='xy', scale=1, color='black', alpha=0.25)

# Labeling
ax.set_xlabel('Probability of Player 1 choosing Action 1')
ax.set_ylabel('Probability of Player 2 choosing Action 1')
ax.set_title('Replicator Dynamics for Prisoners Dilemma - Epsilon-Greedy Q-Learning')
ax.grid(True)

# Simulate learning trajectories and plot them
for q_table in start_points:
    epsilon = 0.2
    alpha = 0.002
    gamma = 0.7

    # Create instances of RLAgent for both players
    agent1 = EpsilonGreedyQLearningAgent(num_states=num_states, num_actions=num_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)
    agent2 = EpsilonGreedyQLearningAgent(num_states=num_states, num_actions=num_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 10000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    ax.scatter(q_learning_probs_0_for_action_0[0], q_learning_probs_1_for_action_0[0], s=25, color = colors[colors_i])

    # Plot the trajectory on the same axes as the quiver plot
    ax.plot(q_learning_probs_0_for_action_0, q_learning_probs_1_for_action_0, linestyle='-', color=colors[colors_i], alpha=1.0)
    colors_i += 1

# Display the plot
plt.show()

"""Boltzmann Q-Learning"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/new-ml-project-kul-master

from Assets.Game import Game
from Assets.Graph import Graph

legal_games = ["rock_paper_scissors", "subsidy_game", "Battle_of_the_sexes", "prisoners_dilemma"]
game_name = "prisoners_dilemma"
alpha = 0.001
tau = 0.5
game = Game(game_name)
kappa = 1

fig = Graph.compute_vector_field(game, alpha, tau , kappa=kappa, normalise= True)

# title = f"{game.get_game_name()}\n"
# title += f"| alpha={alpha}, tau={tau}, kappa={kappa}"

fig.update_layout(xaxis_title='Probability of Player 1 picking Action 1',
                yaxis_title='Probability of Player 2 picking Action 1',
                # legend_title='Traces',
                xaxis=dict(range=[0, 1], scaleanchor="y"),  # Set X axis range and link scale to Y axis
                yaxis=dict(range=[0, 1], scaleratio=1),     # Set Y axis range and ensure square aspect ratio
                width=600,  # Set the width of the plot
                height=600)  # Set the height of the plot to the same value as width to make it square

# Show the plot
fig.show()

import plotly.graph_objects as go
import numpy as np

# Parameters
num_states = 1
num_actions = 2

# Generate random start points for learning trajectories
start_points = [1 * np.random.rand(num_states, num_actions) - 0.5 for _ in range(10)]

colors_i = 0

payoff_matrix = prisoners_dilemma_payoffs + 4

for q_table in start_points:
    temperature = 10
    alpha = 0.001
    gamma = 0.9
    leniency = 1

    # Create instances of Lenient Boltzmann Q Learning Agent
    agent1 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                            temperature=temperature, alpha=alpha, gamma=gamma, leniency=leniency)
    agent2 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                            temperature=temperature, alpha=alpha, gamma=gamma, leniency=leniency)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 10000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    # Add initial scatter point
    fig.add_trace(go.Scatter(
        x=[q_learning_probs_0_for_action_0[0]],
        y=[q_learning_probs_1_for_action_0[0]],
        mode='markers',
        marker=dict(size=8, color=colors[colors_i]),
        # name=f"Start {colors_i+1}"
    ))

    # Plot the trajectory on the same figure
    fig.add_trace(go.Scatter(
        x=q_learning_probs_0_for_action_0[::10],
        y=q_learning_probs_1_for_action_0[::10],
        mode='lines',
        line=dict(color=colors[colors_i], width=2),
        # name=f"Trajectory {colors_i+1}"
    ))

    colors_i += 1

fig.show()

payoff_matrix = prisoners_dilemma_payoffs + 4

temperature = 0.5

# Define the Lenient Boltzmann Q-learning adjusted replicator dynamics
def replicator_dynamics(p1, p2, temperature):
    # Expected payoffs for Player 1
    pi1_a1 = p2 * payoff_matrix[0, 0, 0] + (1 - p2) * payoff_matrix[0, 1, 0]
    pi1_a2 = p2 * payoff_matrix[1, 0, 0] + (1 - p2) * payoff_matrix[1, 1, 0]
    avg_pi1 = p1 * pi1_a1 + (1 - p1) * pi1_a2

    # Expected payoffs for Player 2
    pi2_a1 = p1 * payoff_matrix[0, 0, 1] + (1 - p1) * payoff_matrix[1, 0, 1]
    pi2_a2 = p1 * payoff_matrix[0, 1, 1] + (1 - p1) * payoff_matrix[1, 1, 1]
    avg_pi2 = p2 * pi2_a1 + (1 - p2) * pi2_a2

    # Replicator equations
    dp1 = p1 * (pi1_a1 - avg_pi1)
    dp2 = p2 * (pi2_a1 - avg_pi2)

    return dp1, dp2

# Grid of probabilities
p1_vals = np.linspace(0, 1, 20)
p2_vals = np.linspace(0, 1, 20)
P1, P2 = np.meshgrid(p1_vals, p2_vals)

# Compute the derivatives
dP1, dP2 = np.zeros(P1.shape), np.zeros(P2.shape)
for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        dp1, dp2 = replicator_dynamics(P1[i, j], P2[i, j], temperature)
        dP1[i, j] = dp1
        dP2[i, j] = dp2

# Plot the phase plot
plt.figure(figsize=(8, 8))
plt.quiver(P1, P2, dP1/10.0, dP2/10.0, angles='xy', scale_units='xy', scale=1, color='black')
plt.xlabel('Probability of Player 1 choosing Action 1')
plt.ylabel('Probability of Player 2 choosing Action 1')
plt.title('Replicator Dynamics for Prisoners Dilemma - for Boltzmann Q-Learning')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_states = 1
num_actions = 2

# Generate random start points for learning trajectories
start_points = [np.random.rand(num_states, num_actions) - 0.5 for _ in range(10)]
colors = plt.cm.get_cmap('hsv', len(start_points))
colors = [colors(i) for i in range(len(start_points))]
colors_i = 0

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 8))

 
# The quiver plot should show the replicator dynamics vector field
ax.quiver(P1, P2, dP1/10.0, dP2/10.0, angles='xy', scale_units='xy', scale=1, color='black', alpha=0.25)

# Labeling
ax.set_xlabel('Probability of Player 1 choosing Action 1')
ax.set_ylabel('Probability of Player 2 choosing Action 1')
ax.set_title('Replicator Dynamics for Prisoners Dilemma - Boltzmann Q-Learning')
ax.grid(True)

# Simulate learning trajectories and plot them
for q_table in start_points:
    temperature = 0.5
    alpha = 0.001
    gamma = 0.9

    # Create instances of Boltzmann Q Learning Agent
    agent1 = BoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                              temperature=temperature, alpha=alpha, gamma=gamma)
    agent2 = BoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                            temperature=temperature, alpha=alpha, gamma=gamma)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 10000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    ax.scatter(q_learning_probs_0_for_action_0[0], q_learning_probs_1_for_action_0[0], s=25, color = colors[colors_i])

    # Plot the trajectory on the same axes as the quiver plot
    ax.plot(q_learning_probs_0_for_action_0, q_learning_probs_1_for_action_0, linestyle='-', color=colors[colors_i], alpha=1.0)
    colors_i += 1

# Display the plot
plt.show()

"""Lenient Boltzmann Q Learning"""

# Parameters
num_states = 1
num_actions = 2

# Generate random start points for learning trajectories
start_points = [np.random.rand(num_states, num_actions) - 0.5 for _ in range(10)]
colors = plt.cm.get_cmap('hsv', len(start_points))
colors = [colors(i) for i in range(len(start_points))]
colors_i = 0

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 8))

 
# The quiver plot should show the replicator dynamics vector field
ax.quiver(P1, P2, dP1/10.0, dP2/10.0, angles='xy', scale_units='xy', scale=1, color='black', alpha=0.25)

# Labeling
ax.set_xlabel('Probability of Player 1 choosing Action 1')
ax.set_ylabel('Probability of Player 2 choosing Action 1')
ax.set_title('Replicator Dynamics for Prisoners Dilemma - Lenient Boltzmann Q-Learning')
ax.grid(True)

# Simulate learning trajectories and plot them
for q_table in start_points:
    temperature = 0.5
    alpha = 0.001
    gamma = 0.9
    leniency = 5

    # Create instances of Lenient Boltzmann Q Learning Agent
    agent1 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                              temperature=temperature, alpha=alpha, gamma=gamma, leniency = leniency)
    agent2 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                              temperature=temperature, alpha=alpha, gamma=gamma, leniency = leniency)


    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 10000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    ax.scatter(q_learning_probs_0_for_action_0[0], q_learning_probs_1_for_action_0[0], s=25, color = colors[colors_i])

    # Plot the trajectory on the same axes as the quiver plot
    ax.plot(q_learning_probs_0_for_action_0, q_learning_probs_1_for_action_0, linestyle='-', color=colors[colors_i], alpha=1.0)
    colors_i += 1

# Display the plot
plt.show()

"""Lenient Frequency Adjusted Q Learning"""

import numpy as np
import plotly.graph_objects as go
from Assets.Game import Game
from Assets.Graph import Graph

# Define the game and parameters
legal_games = ["rock_paper_scissors", "subsidy_game", "Battle_of_the_sexes", "prisoners_dilemma"]
game_name = "prisoners_dilemma"
alpha = 0.02
tau = 0.01
game = Game(game_name)
kappa = 5

# Compute the vector field
fig = Graph.compute_vector_field(game, alpha, tau, kappa=kappa, normalise=True)

# Layout updates for the vector field plot
fig.update_layout(
    xaxis_title='Probability of Player 1 picking Action 1',
    yaxis_title='Probability of Player 2 picking Action 1',
    xaxis=dict(range=[0, 1], scaleanchor="y"),  # Set X axis range and link scale to Y axis
    yaxis=dict(range=[0, 1], scaleratio=1),     # Set Y axis range and ensure square aspect ratio
    width=600,  # Set the width of the plot
    height=600  # Set the height of the plot to the same value as width to make it square
)

# Generate random start points for learning trajectories
num_states = 1
num_actions = 2
start_points = [2 * np.random.rand(num_states, num_actions) - 1 for _ in range(10)]
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'brown']
colors_i = 0

payoff_matrix = prisoners_dilemma_payoffs + 4

# Simulate learning trajectories and plot them
for q_table in start_points[:4]:
    # Parameters for Lenient Frequency Adjusted Q-learning
    beta = 1.5
    temperature = 0.01
    alpha = 0.02
    gamma = 0.9
    leniency = 5

    # Create instances of Lenient Frequency Adjusted Q Learning Agent
    agent1 = LenientFrequencyAdjustedQLearningAgent(num_states=1, num_actions=num_actions, leniency=leniency, temperature=temperature, alpha=alpha, gamma=gamma, beta=beta)
    agent2 = LenientFrequencyAdjustedQLearningAgent(num_states=1, num_actions=num_actions, leniency=leniency, temperature=temperature, alpha=alpha, gamma=gamma, beta=beta)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 100000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, False)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_1_for_action_0 = [prob[0][0] for prob in probs_1]

    # Add initial scatter point
    fig.add_trace(go.Scatter(
        x=[q_learning_probs_0_for_action_0[0]],
        y=[q_learning_probs_1_for_action_0[0]],
        mode='markers',
        marker=dict(size=8, color=colors[colors_i]),
        name=f"Start {colors_i+1}"
    ))

    # Plot the trajectory on the same figure
    fig.add_trace(go.Scatter(
        x=q_learning_probs_0_for_action_0,
        y=q_learning_probs_1_for_action_0,
        mode='lines',
        line=dict(color=colors[colors_i], width=2),
        name=f"Trajectory {colors_i+1}"
    ))

    colors_i += 1

# Show the plot with learning trajectories overlaying the vector field
fig.show()

"""## Rock Paper Scissors

Epsilon Greedy Q Learning
"""

payoff_matrix = rock_paper_scissors_payoffs

def replicator_dynamics(p, payoff_matrix):
    """Calculate the replicator dynamics for a given strategy profile."""
    p1, p2, p3 = p
    fitness = np.dot(payoff_matrix, p)
    avg_fitness = np.dot(fitness, p)
    dp = p * (fitness - avg_fitness)
    return dp

# Generate a triangular grid
t, l, r = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 10), indexing='ij')
t = t.flatten()
l = l.flatten()
r = r.flatten()
# Ensure points are within the triangle
valid_indices = (t + l <= 1) & (t >= 0) & (l >= 0) & (r >= 0)
t = t[valid_indices]
l = l[valid_indices]
r = r[valid_indices]

# Compute the dynamics
dt, dl, dr = np.zeros_like(t), np.zeros_like(l), np.zeros_like(r)
for i in range(len(t)):
    p = np.array([t[i], l[i], r[i]])
    dp = replicator_dynamics(p, payoff_matrix)
    dt[i], dl[i], dr[i] = dp[0], dp[1], dp[2]

# Normalize vector lengths for better visualization
length = np.sqrt(dt ** 2 + dl ** 2 + dr ** 2)
dt /= length
dl /= length
dr /= length

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='ternary'))
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

# Plot vector field
pc = ax.quiver(t, l, r, dt/20.0, dl/20.0, dr/20.0, angles='xy', scale_units='xy', scale=1)
ax.set_title("Replicator Dynamics for Rock Paper Scissors - Epislon Greedy Q-Learning")

plt.show()

# Parameters
num_states = 1
num_actions = 3

# Generate random start points for learning trajectories
start_points = [2 * np.random.rand(num_states, num_actions) - 1.0 for _ in range(5)]
colors = plt.cm.get_cmap('hsv', len(start_points))
colors = [colors(i) for i in range(len(start_points))]
colors_i = 0

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='ternary'))
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

# The quiver plot should show the replicator dynamics vector field
ax.quiver(t, l, r, dt/20.0, dl/20.0, dr/20.0, angles='xy', scale_units='xy', scale=1, alpha = 0.2)
ax.set_title("Replicator Dynamics for Rock Paper Scissors - Epislon Greedy Q-Learning")
ax.grid(True)

# Simulate learning trajectories and plot them
for q_table in start_points:
    epsilon = 0.25
    alpha = 0.01
    gamma = 0.9

    # Create instances of RLAgent for both players
    agent1 = EpsilonGreedyQLearningAgent(num_states=num_states, num_actions=num_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)
    agent2 = EpsilonGreedyQLearningAgent(num_states=num_states, num_actions=num_actions, epsilon=epsilon, alpha=alpha, gamma=gamma)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 2000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, True)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_0_for_action_1 = [prob[0][1] for prob in probs_0]
    q_learning_probs_0_for_action_2 = [prob[0][2] for prob in probs_0]
    ax.scatter(q_learning_probs_0_for_action_0[0], q_learning_probs_0_for_action_1[0], q_learning_probs_0_for_action_2[0], s=25, color = colors[colors_i])

    # Plot the trajectory on the same axes as the quiver plot
    ax.plot(q_learning_probs_0_for_action_0, q_learning_probs_0_for_action_1, q_learning_probs_0_for_action_2, linestyle='-', color=colors[colors_i], alpha=1.0)
    colors_i += 1

# Display the plot
plt.show()

"""Boltzmann Q Learning"""

payoff_matrix = rock_paper_scissors_payoffs

def boltzmann_dynamics(p, payoff_matrix, temperature=1.0):
    """Calculate the Boltzmann Q-learning dynamics for a given strategy profile."""
    p1, p2, p3 = p
    # Compute Q-values based on the Boltzmann distribution
    q_values = np.dot(payoff_matrix, p)
    exp_q = np.exp(q_values / temperature)
    probabilities = exp_q / np.sum(exp_q)

    # Expected utility
    avg_utility = np.dot(probabilities, q_values)

    # Compute dynamics (gradient)
    dp = p * (q_values - avg_utility)
    return dp

# Generate a triangular grid
t, l, r = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 10), indexing='ij')
t = t.flatten()
l = l.flatten()
r = r.flatten()
# Ensure points are within the triangle
valid_indices = (t + l <= 1) & (t >= 0) & (l >= 0) & (r >= 0)
t = t[valid_indices]
l = l[valid_indices]
r = r[valid_indices]

# Compute the dynamics
dt, dl, dr = np.zeros_like(t), np.zeros_like(l), np.zeros_like(r)
temperature = 0.5
for i in range(len(t)):
    p = np.array([t[i], l[i], r[i]])
    dp = boltzmann_dynamics(p, payoff_matrix, temperature)
    dt[i], dl[i], dr[i] = dp[0], dp[1], dp[2]

# Normalize vector lengths for better visualization
length = np.sqrt(dt ** 2 + dl ** 2 + dr ** 2)
length[length == 0] = 1  # Avoid division by zero for vectors with zero length
dt /= length
dl /= length
dr /= length

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='ternary'))
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

# Plot vector field
pc = ax.quiver(t, l, r, dt/20.0, dl/20.0, dr/20.0, color='black', angles='xy', scale_units='xy', scale=1)
ax.set_title("Replicator Dynamics for Rock Paper Scissors - Boltzmann Q-Learning")

plt.show()

# Parameters
num_states = 1
num_actions = 3

# Generate random start points for learning trajectories
start_points = [2 * np.random.rand(num_states, num_actions) - 1.0 for _ in range(10)]
colors = plt.cm.get_cmap('hsv', len(start_points))
colors = [colors(i) for i in range(len(start_points))]
colors_i = 0

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='ternary'))
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

# The quiver plot should show the replicator dynamics vector field
ax.quiver(t, l, r, dt/20.0, dl/20.0, dr/20.0, angles='xy', scale_units='xy', scale=1, alpha = 0.2)
ax.set_title("Replicator Dynamics for Rock Paper Scissors - Boltzmann Q-Learning")
ax.grid(True)

# Simulate learning trajectories and plot them
for q_table in start_points:
    temperature = 0.75
    alpha = 0.005
    gamma = 0.9

    # Create instances of Boltzmann Q Learning Agent
    agent1 = BoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                              temperature=temperature, alpha=alpha, gamma=gamma)
    agent2 = BoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                            temperature=temperature, alpha=alpha, gamma=gamma)
    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 10000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, True)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_0_for_action_1 = [prob[0][1] for prob in probs_0]
    q_learning_probs_0_for_action_2 = [prob[0][2] for prob in probs_0]

    ax.scatter(q_learning_probs_0_for_action_0[0], q_learning_probs_0_for_action_1[0], q_learning_probs_0_for_action_2[0], s=25, color = colors[colors_i])

    # Plot the trajectory on the same axes as the quiver plot
    ax.plot(q_learning_probs_0_for_action_0, q_learning_probs_0_for_action_1, q_learning_probs_0_for_action_2, linestyle='-', color=colors[colors_i], alpha=1.0)
    colors_i += 1

# Display the plot
plt.show()

"""Lenient Boltzmann Q Learning"""

# Parameters
num_states = 1
num_actions = 3

# Generate random start points for learning trajectories
start_points = [2 * np.random.rand(num_states, num_actions) - 1.0 for _ in range(5)]
colors = plt.cm.get_cmap('hsv', len(start_points))
colors = [colors(i) for i in range(len(start_points))]
colors_i = 0

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='ternary'))
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

# The quiver plot should show the replicator dynamics vector field
ax.quiver(t, l, r, dt/20.0, dl/20.0, dr/20.0, angles='xy', scale_units='xy', scale=1, alpha = 0.2)
ax.set_title("Replicator Dynamics for Rock Paper Scissors - Lenient Boltzmann Q-Learning")
ax.grid(True)

# Simulate learning trajectories and plot them
for q_table in start_points:
    leniency = 5
    temperature = 0.75
    alpha = 0.001
    gamma = 0.9

    # Create instances of Lenient Boltzmann Q Learning Agent
    agent1 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                              temperature=temperature, alpha=alpha, gamma=gamma, leniency = leniency)
    agent2 = LenientBoltzmannQLearningAgent(num_states=1, num_actions=num_actions,
                                              temperature=temperature, alpha=alpha, gamma=gamma, leniency = leniency)
    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 20000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, True)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_0_for_action_1 = [prob[0][1] for prob in probs_0]
    q_learning_probs_0_for_action_2 = [prob[0][2] for prob in probs_0]

    ax.scatter(q_learning_probs_0_for_action_0[0], q_learning_probs_0_for_action_1[0], q_learning_probs_0_for_action_2[0], s=25, color = colors[colors_i])

    # Plot the trajectory on the same axes as the quiver plot
    ax.plot(q_learning_probs_0_for_action_0, q_learning_probs_0_for_action_1, q_learning_probs_0_for_action_2, linestyle='-', color=colors[colors_i], alpha=1.0)
    colors_i += 1

# Display the plot
plt.show()

"""Lenient Frequency Adjusted Q Learning"""

def frequency_adjusted_boltzmann_dynamics(p, payoff_matrix, temperature=1.0, beta=0.1, frequency_table=None):
    """Calculate the Frequency Adjusted Boltzmann Q-learning dynamics for a given strategy profile."""
    p1, p2, p3 = p
    # Compute Q-values based on the Boltzmann distribution
    q_values = np.dot(payoff_matrix, p)
    exp_q = np.exp(q_values / temperature)
    probabilities = exp_q / np.sum(exp_q)

    # Expected utility
    avg_utility = np.dot(probabilities, q_values)

    # Frequency adjustment
    if frequency_table is None:
        frequency_table = np.ones(len(p))  # If no frequency table is provided, assume all frequencies are 1

    freq_adjustment = np.minimum(1, beta / frequency_table)

    # Compute dynamics (gradient) with frequency adjustment
    dp = freq_adjustment * p * (q_values - avg_utility)
    return dp

# Generate a triangular grid
t, l, r = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, 1, 10), indexing='ij')
t = t.flatten()
l = l.flatten()
r = r.flatten()
# Ensure points are within the triangle
valid_indices = (t + l + r <= 1) & (t >= 0) & (l >= 0) & (r >= 0)
t = t[valid_indices]
l = l[valid_indices]
r = r[valid_indices]

# Initialize frequency table with ones (this would typically be updated during learning)
frequency_table = np.ones(3)

# Compute the dynamics
dt, dl, dr = np.zeros_like(t), np.zeros_like(l), np.zeros_like(r)
temperature = 0.5
beta = 0.001
for i in range(len(t)):
    p = np.array([t[i], l[i], r[i]])
    dp = frequency_adjusted_boltzmann_dynamics(p, payoff_matrix, temperature, beta, frequency_table)
    dt[i], dl[i], dr[i] = dp[0], dp[1], dp[2]

# Normalize vector lengths for better visualization
length = np.sqrt(dt ** 2 + dl ** 2 + dr ** 2)
length[length == 0] = 1  # Avoid division by zero for vectors with zero length
dt /= length
dl /= length
dr /= length

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='ternary'))
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

# Plot vector field
pc = ax.quiver(t, l, r, dt/20.0, dl/20.0, dr/20.0, color='black', angles='xy', scale_units='xy', scale=1)
ax.set_title("Replicator Dynamics for Rock Paper Scissors - Lenient Frequency Adjusted Q-Learning")

plt.show()

# Parameters
num_states = 1
num_actions = 3

# Generate random start points for learning trajectories
start_points = [2 * np.random.rand(num_states, num_actions) - 1 for _ in range(10)]
colors = plt.cm.get_cmap('hsv', len(start_points))
colors = [colors(i) for i in range(len(start_points))]
colors_i = 0

# # Plot the results
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='ternary'))
fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)

# The quiver plot should show the replicator dynamics vector field
ax.quiver(t, l, r, dt/20.0, dl/20.0, dr/20.0, angles='xy', scale_units='xy', scale=1, alpha = 0.2)
ax.set_title("Replicator Dynamics for Rock Paper Scissors - Lenient Frequency Adjusted Q-Learning")
ax.grid(True)

# Simulate learning trajectories and plot them
for q_table in start_points:
    # For Lenient Frequency Adjusted Q-learning
    leniency = 10
    beta = 0.5
    temperature = 100
    alpha = 0.1
    gamma = 0.9

    # Create instances of Lenient Frequency Adjusted Q Learning Agent
    agent1 = LenientFrequencyAdjustedQLearningAgent(num_states=1, num_actions=num_actions, leniency = leniency, temperature = temperature, alpha = alpha, gamma = gamma, beta = beta)
    agent2 = LenientFrequencyAdjustedQLearningAgent(num_states=1, num_actions=num_actions, leniency = leniency, temperature = temperature, alpha = alpha, gamma = gamma, beta = beta)

    agent1.set_q_table(q_table)

    # Play multiple episodes and get probabilities
    num_episodes = 100000
    rewards_q_learning, probs_0, probs_1 = play_multiple_episodes(agent1, agent2, num_episodes, payoff_matrix, True)

    # Extract probabilities of choosing Action 0 for both agents
    q_learning_probs_0_for_action_0 = [prob[0][0] for prob in probs_0]
    q_learning_probs_0_for_action_1 = [prob[0][1] for prob in probs_0]
    q_learning_probs_0_for_action_2 = [prob[0][2] for prob in probs_0]

    ax.scatter(q_learning_probs_0_for_action_0[0], q_learning_probs_0_for_action_1[0], q_learning_probs_0_for_action_2[0], s=25, color = colors[colors_i])
    # Plot the trajectory on the same axes as the quiver plot
    ax.plot(q_learning_probs_0_for_action_0, q_learning_probs_0_for_action_1, q_learning_probs_0_for_action_2, linestyle='-', color=colors[colors_i], alpha=1.0)
    colors_i += 1

# Display the plot
plt.show()
