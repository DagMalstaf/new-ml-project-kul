# File: train_advanced_nn_mcts_agent.py
import os
import datetime
import yaml
import logging
import pyspiel
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from task4.agent.advanced_nn_mcts_agent import create_advanced_nn_mcts_agent


BLUE = "\033[1;34m"
RESET = "\033[0m"

logging.basicConfig(level=logging.INFO, format=f'{BLUE}%(asctime)s - %(levelname)s - %(message)s{RESET}')


def run_self_play(agent, game, num_episodes, strategy, num_gpus):
    trajectories = []
    for i in range(num_episodes):
        logging.info(f"Running episode: {i} out of {num_episodes} episodes.")
        state = game.new_initial_state()
        trajectory = []
        while not state.is_terminal():
            action = agent.get_action(state, strategy, training=True)
            trajectory.append((state.clone(), action))
            state.apply_action(action)
        if len(trajectory) == 0:
            logging.warning(f"Episode {i} generated an empty trajectory. Possible early termination or error in game logic.")
        else:
            trajectories.append(trajectory)

    if len(trajectories) < num_gpus:
        error_message = f"Insufficient trajectories for GPU distribution: required at least {num_gpus}, but got {len(trajectories)}"
        logging.error(error_message)
        raise ValueError(error_message)

    return trajectories # list(state, action)


def process_trajectory(trajectories):
    inputs = []
    policy_targets = []
    value_targets = []
    for trajectory in trajectories:
        for state, action in trajectory:
            inputs.append(np.array(state.observation_tensor()))
            num_actions = state.num_distinct_actions()
            policy_target = np.zeros(num_actions)
            policy_target[action] = 1
            value_target = state.returns()[0] 
            policy_targets.append(policy_target)
            value_targets.append(value_target)
    return np.array(inputs), np.array(policy_targets), np.array(value_targets)


def train_advanced_nn(game_name, num_simulations, exploration_coefficient, num_episodes, batch_size=16, validation_split=0.1):
    strategy = get_distribution_strategy()
    print(f'Using strategy with devices: {strategy.num_replicas_in_sync}')

    with strategy.scope():
        game = pyspiel.load_game(game_name)
        agent = create_advanced_nn_mcts_agent(game, num_simulations, exploration_coefficient)

        num_gpus = strategy.num_replicas_in_sync  
        min_required_batch_size = num_gpus * 16 

        # This check does not fail
        if batch_size < min_required_batch_size:
            batch_size = min_required_batch_size
            logging.info(f"Adjusted batch size to {batch_size} due to GPU requirements.")


        logging.info(f"Running self play \n")
        trajectories = run_self_play(agent, game, num_episodes, strategy, num_gpus)

        if len(trajectories) == 0:
            raise ValueError("No valid trajectories generated. Check game logic or parameters.")
        
        logging.info(f"Processing trajectories. \n")
        
        inputs, policy_targets, value_targets = process_trajectory(trajectories)

        required_batch_size = strategy.num_replicas_in_sync
        if len(inputs) % required_batch_size != 0:
            adjusted_batch_size = (len(inputs) // required_batch_size) * required_batch_size
            logging.warning(f"Adjusting batch size from {batch_size} to {adjusted_batch_size} to fit GPU requirements.")
            batch_size = adjusted_batch_size
        
        logging.info(f"Total input size: {len(inputs)}, Batch size: {batch_size}, Number of GPUs: {strategy.num_replicas_in_sync}")
        if len(inputs) < batch_size * strategy.num_replicas_in_sync:
            raise ValueError("Not enough data to distribute across GPUs properly.")


        if len(inputs) == 0:
            raise ValueError("Processed data resulted in an empty dataset.")
        
        if len(inputs) % batch_size != 0:
            logging.info(f"Adjusting batch size from {batch_size} due to dataset size.")
            batch_size = len(inputs) // (len(inputs) // batch_size)  


        train_inputs, train_policy_targets, train_value_targets, val_inputs, val_policy_targets, val_value_targets = split_dataset(inputs, policy_targets, value_targets)
        
        if len(train_inputs) == 0 or len(val_inputs) == 0:
                raise ValueError("Training or validation dataset is empty. Check your data split or preprocessing.")

        logging.info(train)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, {'policy_output': train_policy_targets, 'value_output': train_value_targets}))
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, {'policy_output': val_policy_targets, 'value_output': val_value_targets}))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset = train_dataset.repeat()
        val_dataset = val_dataset.repeat()

        steps_per_epoch = len(train_inputs) // batch_size
        validation_steps = len(val_inputs) // batch_size

        if steps_per_epoch == 0 or validation_steps == 0:
                raise ValueError("Not enough data for the number of steps per epoch or validation. Adjust the batch size or dataset size.")

        log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        callbacks = [
            ModelCheckpoint(f'task4/models/advanced-nn-mcts-models/long-train-7x7-save.keras', save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001, verbose=1),
            tensorboard_callback
        ]

        history = agent.neural_network.fit(
            train_dataset,
            epochs=50,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            callbacks=callbacks,
            validation_steps=validation_steps
        )

        agent.neural_network.save(f'task4/models/advanced-nn-mcts-models/long-train-7x7.keras')



def split_dataset(inputs, policy_targets, value_targets, validation_split=0.1):
    num_samples = len(inputs)
    num_val_samples = int(validation_split * num_samples)
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    inputs = inputs[indices]
    policy_targets = policy_targets[indices]
    value_targets = value_targets[indices]
    
    val_inputs = inputs[:num_val_samples]
    train_inputs = inputs[num_val_samples:]
    val_policy_targets = policy_targets[:num_val_samples]
    val_value_targets = value_targets[:num_val_samples]
    train_policy_targets = policy_targets[num_val_samples:]
    train_value_targets = value_targets[num_val_samples:]

    return train_inputs, train_policy_targets, train_value_targets, val_inputs, val_policy_targets, val_value_targets


def get_distribution_strategy():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logging.error("Failed to set memory growth: {}".format(e))

            return tf.distribute.MirroredStrategy()
        else:
            return tf.distribute.get_strategy()
    except RuntimeError as e:
        print(e)
        return tf.distribute.get_strategy()
