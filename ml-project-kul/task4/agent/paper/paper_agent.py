import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from paper import DeepStrategyNetwork, DeepValueNetwork, AdvancedMCTS_NNAgent

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = self.get_legal_actions()
        self.value_net_evaluation = None

    def get_legal_actions(self):
        return self.state.get_possible_actions() if not self.is_terminal_node() else []

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self, strategy_network):
        current_state = self.state
        while not current_state.is_game_over():
            possible_actions = current_state.get_possible_actions()
            nn_input = current_state.to_nn_input()
            action_probabilities = strategy_network.predict(nn_input)[0]  # Assuming the output shape includes the batch dimension
            # Ensure probabilities are only for legal actions
            probabilities = np.zeros_like(action_probabilities)
            probabilities[possible_actions] = np.exp(action_probabilities[possible_actions] - np.max(action_probabilities[possible_actions]))
            probabilities /= np.sum(probabilities)  # Normalize to get a proper probability distribution
            action = np.random.choice(len(action_probabilities), p=probabilities)  # Choose action based on these probabilities
            current_state = current_state.move(action)
        return current_state.get_result()


    def backpropagate(self, result, value_network):
        self.visits += 1
        nn_input = self.state.to_nn_input()
        value_prediction = value_network.predict(nn_input)[0][0]  # Assuming the output is a single scalar
        self.wins += result * value_prediction  # Adjust based on your game's scoring and outcome system
        if self.parent:
            self.parent.backpropagate(result, value_network)


class MCTS:
    def __init__(self, value_network: Model, strategy_network: Model, number_of_simulations=100, exploration_weight=1.41):
        self.value_network = value_network
        self.strategy_network = strategy_network
        self.exploration_weight = exploration_weight
        self.number_of_simulations = number_of_simulations

    def search(self, initial_state):
        root = Node(initial_state)
        for _ in range(self.number_of_simulations):
            node = root
            while node.untried_actions == [] and node.children != []:
                node = self.select_child(node)
            if node.untried_actions != []:
                action = np.random.choice(node.untried_actions)
                node = self.expand(node, action)
            result = node.rollout(self.strategy_network)
            node.backpropagate(result, self.value_network)
        return self.get_best_move(root)

    def select_child(self, node):
        best_value = float('-inf')
        best_nodes = []
        for child in node.children:
            ucb1 = self.calculate_ucb1(child)
            if ucb1 > best_value:
                best_value = ucb1
                best_nodes = [child]
            elif ucb1 == best_value:
                best_nodes.append(child)
        return np.random.choice(best_nodes)

    def calculate_ucb1(self, node):
        win_rate = node.wins / node.visits
        exploration_factor = np.sqrt(np.log(node.parent.visits) / node.visits)
        return win_rate + self.exploration_weight * exploration_factor

    def expand(self, node, action):
        new_state = node.state.move(action)
        new_node = Node(new_state, node)
        node.children.append(new_node)
        node.untried_actions.remove(action)
        return new_node

    def get_best_move(self, node):
        best_move = None
        best_score = -float('inf')
        for child in node.children:
            if child.visits > best_score:
                best_score = child.visits
                best_move = child
        return best_move.state.get_last_move()

class AdvancedMCTS_NNAgent:
    def __init__(self, value_network, strategy_network):
        self.mcts = MCTS(value_network, strategy_network)

    def choose_action(self, state, turn):
        if turn < 10:
            return self.alpha_beta_action(state)
        else:
            return self.mcts_action(state)

    def mcts_action(self, state):
        return self.mcts.search(state)

    def alpha_beta_action(self, state):
        return self.alpha_beta_pruning(Node(state), depth=10, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)

    def alpha_beta_pruning(self, node, depth, alpha, beta, maximizing_player):
        if depth == 0 or node.is_terminal_node():
            return self.evaluate_state(node.state)
        
        if maximizing_player:
            max_eval = float('-inf')
            for action in node.get_legal_actions():
                child_state = node.state.move(action)
                eval = self.alpha_beta_pruning(Node(child_state, node), depth-1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for action in node.get_legal_actions():
                child_state = node.state.move(action)
                eval = self.alpha_beta_pruning(Node(child_state, node), depth-1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_state(self, state):
        nn_input = state.to_nn_input()
        predicted_value = self.value_network.predict(nn_input)
        return predicted_value[0][0]  # Assuming the network outputs a scalar value per input


class GameState:
    def to_nn_input(self):
        return np.random.rand(1, 11, 11, 17)  # Shape must match the input shape expected by the network




def create_paper_agent(input_shape, num_actions, strategy_model_path=None, value_model_path=None, num_simulations=100, exploration_coefficient=1.41):
    logging.info("Initializing Paper Agent")

    if strategy_model_path:
        strategy_network = load_model(strategy_model_path)
        logging.info("Loaded strategy network from {}".format(strategy_model_path))
    else:
        strategy_network = DeepStrategyNetwork(input_shape=input_shape, num_actions=num_actions)
        strategy_network.compile(optimizer='adam', loss='categorical_crossentropy')
        logging.info("Initialized new strategy network")

    # Initialize or load the value network
    if value_model_path:
        value_network = load_model(value_model_path)
        logging.info("Loaded value network from {}".format(value_model_path))
    else:
        value_network = DeepValueNetwork(input_shape=input_shape)
        value_network.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("Initialized new value network")

    # Create the MCTS NN Agent with the loaded or initialized networks
    agent = AdvancedMCTS_NNAgent(value_network=value_network, strategy_network=strategy_network)
    agent.mcts = MCTS(value_network=value_network, strategy_network=strategy_network, number_of_simulations=num_simulations, exploration_weight=exploration_coefficient)
    
    logging.info("Agent created with simulation count {} and exploration coefficient {}".format(num_simulations, exploration_coefficient))
    return agent
