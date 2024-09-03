# agent/nn_mcts_agent.py
import logging
import os
import numpy as np
import tensorflow as tf
from task4.agent.mcts.mcts_agent_V0 import MCTSAgent
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
from pyspiel import load_game


BLUE = "\033[1;34m"
RESET = "\033[0m"

logging.basicConfig(level=logging.INFO, format=f'{BLUE}%(asctime)s - %(levelname)s - %(message)s{RESET}')


class BasicNNModel(models.Model):
    def __init__(self, num_actions, input_shape, trainable=True, **kwargs):
        super(BasicNNModel, self).__init__(**kwargs)
        self.trainable = trainable
        self.num_actions = num_actions
        self.input_shape = input_shape
        
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.batch_norm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(128, activation='relu')
        self.batch_norm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.5)
        
        self.policy_output = layers.Dense(num_actions, activation='softmax')
        self.value_output = layers.Dense(1, activation='tanh')

    def call(self, inputs):
        x = tf.reshape(inputs, (-1,) + tuple(self.input_shape))  
        x = self.flatten(x)
        x = self.dropout1(self.batch_norm1(self.dense1(x)))
        x = self.dropout2(self.batch_norm2(self.dense2(x)))
        policy = self.policy_output(x)
        value = self.value_output(x)
        return policy, value

    def get_config(self):
        config = super(BasicNNModel, self).get_config()
        config.update({
            "num_actions": self.num_actions,
            "trainable": self.trainable,
            "input_shape": self.input_shape  
        })
        return config


    @classmethod
    def from_config(cls, config):
        num_actions = config.get('num_actions')
        trainable = config.get('trainable', True)
        # Extract the input shape or other necessary parameters here if needed.
        input_shape = config.get('input_shape')  # Make sure to store and retrieve this properly in get_config
        return cls(num_actions=num_actions, input_shape=input_shape, trainable=trainable)


class NN_MCTSAgent(MCTSAgent):
    def __init__(self, game, num_simulations, exploration_coefficient, nn_model, input_shape):
        super().__init__(game, num_simulations, exploration_coefficient)
        self.neural_network = nn_model
        self.input_shape = input_shape

    def _state_to_nn_input(self, state):
        tensor = np.array(state.observation_tensor())
        print(self.input_shape)
        reshaped_tensor = tensor.reshape(self.input_shape)
        return reshaped_tensor


    def get_action(self, state):
        """Determines the best action by combining MCTS and neural network insights."""
        if self.serialize_state(state) not in self.tree:
            self.expand_node(state)
        
        for _ in range(self.num_simulations):
            self.run_simulation(state.clone())

        state_key = self.serialize_state(state)
        most_visited_action = max(self.tree[state_key]['N'], key=self.tree[state_key]['N'].get)
        return most_visited_action
    
    @tf.function(reduce_retracing=True)
    def expand_node(self, state):
        """Expands the node with neural network predictions when it is first visited."""
        nn_input = self._state_to_nn_input(state)
        predictions = self.neural_network(np.expand_dims(nn_input, axis=0), training=False)
        policy_logits, value_estimate = predictions[0], predictions[1]

        policy_logits = tf.squeeze(policy_logits, axis=0)

        legal_actions = state.legal_actions()
        policy = tf.zeros_like(policy_logits)  # Create a zero tensor
        max_logits = tf.reduce_max(policy_logits)
        exp_logits = tf.exp(policy_logits - max_logits)

        # Create the policy only for legal actions
        indices = tf.expand_dims(legal_actions, 1)
        updates = tf.gather(exp_logits, legal_actions)
        policy = tf.tensor_scatter_nd_update(policy, indices, updates)

        # Normalize the policy tensor to sum to 1
        sum_exp_logits = tf.reduce_sum(tf.gather(policy, legal_actions))
        normalized_updates = updates / sum_exp_logits
        policy = tf.tensor_scatter_nd_update(policy, indices, normalized_updates)

        # Store values as tensors in self.tree and handle conversion outside of tf.function if necessary
        serialized_state = self.serialize_state(state)
        self.tree[serialized_state] = {
            'Q': {int(a): 0.0 for a in legal_actions},
            'N': {int(a): 0 for a in legal_actions},
            'W': {int(a): 0.0 for a in legal_actions},
            'P': {int(a): policy[a] for a in legal_actions},  # Store as tensor
            'legal_actions': legal_actions,
            'value_estimate': value_estimate[0]  # Assumed to be a scalar tensor
        }

    # Note: If you need to process these tensors outside of @tf.function,
    # convert to numpy after calling this function.






def create_nn_mcts_agent(game, num_simulations, exploration_coefficient, model_path=None):
    num_actions = game.num_distinct_actions()
    num_rows_input = game.get_parameters()['num_rows'] + 1
    num_cols_input = game.get_parameters()['num_cols'] + 1 
    input_shape = (3, num_rows_input*num_cols_input, 3)
    logging.info("This is the input shape: {}".format(input_shape))

    if model_path:
        nn_model = load_model(model_path, custom_objects={'BasicNNModel': BasicNNModel})
        nn_model.input_shape = input_shape
        nn_model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])

    else:
        nn_model = BasicNNModel(num_actions, input_shape)
        nn_model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'])

    return NN_MCTSAgent(game, num_simulations, exploration_coefficient, nn_model, input_shape)


def get_agent_for_tournament(player_id):
    base_directory = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))

    model_path = os.path.join(base_directory, 'models', 'basic-nn-mcts-models', 'long-train-7x7.keras')

    game = load_game("dots_and_boxes(num_rows=7,num_cols=7)")
    logging.info("Basic Agent created for tournament with player ID: {}".format(player_id))
    return create_nn_mcts_agent(game, num_simulations=25, exploration_coefficient=1.0, model_path=model_path)











