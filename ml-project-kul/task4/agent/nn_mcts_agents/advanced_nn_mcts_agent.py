# File: agent/advanced_mcts_nn_agent.py
import logging
import os
import yaml
import numpy as np
import tensorflow as tf
from task4.agent.mcts.mcts_agent_V0 import MCTSAgent
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from pyspiel import load_game

RED = "\033[1;31m"
RESET = "\033[0m"

logging.basicConfig(level=logging.DEBUG, format=f'{RED}%(asctime)s - %(levelname)s - %(message)s{RESET}')
import logging

BLUE = "\033[1;34m"
RESET = "\033[0m"

logging.basicConfig(level=logging.INFO, format=f'{BLUE}%(asctime)s - %(levelname)s - %(message)s{RESET}')


class AdvancedNNModel(models.Model):
    def __init__(self, num_actions, input_shape, **kwargs):
        super(AdvancedNNModel, self).__init__(**kwargs)
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.conv1 = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')
        self.batch_norm1 = layers.BatchNormalization()
        self.res_blocks = [self._build_res_block(64) for _ in range(4)]
        self.attention = layers.Attention()
        self.flatten = layers.Flatten()
        self.dense_policy = layers.Dense(128, activation='relu')
        self.policy_output = layers.Dense(num_actions, activation='softmax')
        self.dense_value = layers.Dense(128, activation='relu')
        self.value_output = layers.Dense(1, activation='tanh')

    def _build_res_block(self, num_filters):
        return models.Sequential([
            layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.BatchNormalization()
        ])

    def call(self, inputs):
        x = tf.reshape(inputs, (-1,) + tuple(self.input_shape))

        x = self.conv1(x)
        x = self.batch_norm1(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        shape = tf.shape(x)
        new_shape = (shape[0], shape[1] * shape[2], shape[3])
        x = tf.reshape(x, new_shape)
        x = self.attention([x, x])
        x = self.flatten(x)
        policy = self.policy_output(self.dense_policy(x))
        value = self.value_output(self.dense_value(x))
        return policy, value

    def get_config(self):
        config = super(AdvancedNNModel, self).get_config()
        config.update({
            'num_actions': self.num_actions,
            'input_shape': self.input_shape
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class AdvancedMCTS_NNAgent(MCTSAgent):
    def __init__(self, game, num_simulations, exploration_coefficient, nn_model, input_shape):
        super().__init__(game, num_simulations, exploration_coefficient)
        self.neural_network = nn_model
        self.input_shape = input_shape

    def run_simulation(self, state, strategy=None, training=False):
        path = []

        while not state.is_terminal():
            state_key = self.serialize_state(state)
            if state_key not in self.tree:
                self.populate_tree_with_nn(state, strategy, training=training)
            action, _ = self.select_action(state_key)
            state.apply_action(action)
            path.append((state_key, action))
        reward = state.returns()[0]
        for state_key, action in reversed(path):
            self.tree[state_key]['N'][action] += 1
            self.tree[state_key]['W'][action] += reward
            self.tree[state_key]['Q'][action] = self.tree[state_key]['W'][action] / self.tree[state_key]['N'][action]

    def populate_tree_with_nn(self, state, strategy=None, training=False):
        state_key = self.serialize_state(state)
        nn_input = np.array(state.observation_tensor(), dtype=np.float32)
        
        if nn_input.ndim != 4 or nn_input.shape[1:] != self.input_shape:
            nn_input = nn_input.reshape((1,) + self.input_shape)

        if nn_input.size == 0:
            raise ValueError("Input to the neural network is empty. Check the state observation tensor.")

        if strategy:
            with strategy.scope():
                policy, value = self.neural_network(nn_input, training=training)
        else:
            policy, value = self.neural_network(nn_input, training=training)

        policy = tf.squeeze(policy).numpy()  
        value = value.numpy() 

        legal_actions = state.legal_actions()
        if len(policy) < max(legal_actions, default=-1) + 1:
            raise ValueError("The neural network's output does not cover all legal actions.")

        legal_actions = state.legal_actions()
        policy = policy[legal_actions]
        policy_probabilities = np.exp(policy - np.max(policy))
        policy_probabilities /= np.sum(policy_probabilities)
        self.tree[state_key] = {
            'Q': {a: 0.0 for a in legal_actions},
            'N': {a: 0 for a in legal_actions},
            'W': {a: 0.0 for a in legal_actions},
            'P': {a: policy_probabilities[i] for i, a in enumerate(legal_actions)},
            'V': value,
            'legal_actions': legal_actions
        }

    def select_action(self, state_key, add_exploration=True):
        if state_key not in self.tree:
            raise KeyError("State key not found in the tree: {}".format(state_key))
        node = self.tree[state_key]
        total_visits = sum(node['N'].values()) + 1
        best_action = max(node['legal_actions'], key=lambda a: node['Q'][a] + self.exploration_coefficient * np.sqrt(np.log(total_visits) / (1 + node['N'][a])))
        if add_exploration and np.random.rand() < 0.05:
            best_action = np.random.choice(node['legal_actions'])
        return best_action, node['Q'][best_action]

    def get_action(self, state, strategy=None, training= False):
        for i in range(self.num_simulations):
            self.run_simulation(state.clone(), strategy=strategy, training=training)
        state_key = self.serialize_state(state)
        most_visited_action, _ = self.select_action(state_key, add_exploration=False)
        return most_visited_action
    
def load_config(path):
    """ Load YAML configuration file. """
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def create_advanced_nn_mcts_agent(game, num_simulations, exploration_coefficient, model_path=None):
    logging.info("Game: {}".format(game))
    num_actions = game.num_distinct_actions()
    num_rows_input = game.get_parameters()['num_rows'] + 1
    num_cols_input = game.get_parameters()['num_cols'] + 1 
    input_shape = (3, num_rows_input*num_cols_input, 3)
    custom_objects = {'AdvancedNNModel': AdvancedNNModel}
    if model_path:
        nn_model = load_model(model_path, custom_objects=custom_objects)
    else:
        nn_model = AdvancedNNModel(num_actions=num_actions, input_shape=input_shape)
    nn_model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])
    return AdvancedMCTS_NNAgent(game, num_simulations, exploration_coefficient, nn_model, input_shape=input_shape)

def get_agent_for_tournament(player_id):
    config_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'config.yaml')
    config = load_config(config_path)
    advanced_nn_mcts_config = config['advanced_nn_mcts_agent']

    base_directory = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_directory, 'models', 'advanced-nn-mcts-models', 'long-train-7x7.keras')

    game = load_game("dots_and_boxes")
    agent = create_advanced_nn_mcts_agent(game, advanced_nn_mcts_config['num_simulations'], advanced_nn_mcts_config['exploration_coefficient'], model_path=model_path)
    logging.info("Advanced agent created for tournament with player ID: {}".format(player_id))
    return agent