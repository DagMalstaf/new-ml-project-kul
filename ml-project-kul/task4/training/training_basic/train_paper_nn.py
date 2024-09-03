# File: train_agent.py
import os
import numpy as np
import tensorflow as tf
import pyspiel

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping

from model_definitions import DeepStrategyNetwork, DeepValueNetwork, MCTS, GameState
from simulation_utils import self_play, update_networks

def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('TrainingLogger')







def train_paper_nn(game_name, num_simulations, num_episodes, input_shape):
    logger = setup_logging()
    game = pyspiel.load_game(game_name)
    num_actions = game.

    input_shape = (11, 11, 17)  # Dots and boxes game representation dimensions
    num_actions = 60  # Number of actions for the dots and boxes game

    # Initialize strategy and value networks
    strategy_network = DeepStrategyNetwork(input_shape=input_shape, num_actions=num_actions)
    value_network = DeepValueNetwork(input_shape=input_shape)

    # Compile models with optimizers and loss functions
    strategy_network.compile(optimizer=Adam(learning_rate=0.001), loss=CategoricalCrossentropy())
    value_network.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    # Setup checkpoints and other callbacks
    checkpoint_dir = './model_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(os.path.join(checkpoint_dir, 'strategy_net_best.h5'), save_best_only=True, monitor='loss'),
        ModelCheckpoint(os.path.join(checkpoint_dir, 'value_net_best.h5'), save_best_only=True, monitor='loss'),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=0.00001, verbose=1),
        TensorBoard(log_dir='./logs', update_freq='batch'),
        EarlyStopping(monitor='loss', patience=50, verbose=1)
    ]

    num_episodes = 10000  # Number of self-play games to train on
    save_interval = 100  # Interval for saving model checkpoints

    for episode in range(num_episodes):
        logger.info(f'Starting episode {episode + 1}/{num_episodes}')
        initial_state = GameState()

        # Run a self-play session
        game_data = self_play(initial_state, strategy_network, value_network)

        # Update networks
        update_networks(game_data, strategy_network, value_network, callbacks)

        if episode % save_interval == 0:
            strategy_network.save(f'{checkpoint_dir}/strategy_network_episode_{episode}.h5')
            value_network.save(f'{checkpoint_dir}/value_network_episode_{episode}.h5')
            logger.info(f'Saved models at episode {episode}')

        logger.info(f'Episode {episode + 1} complete')


