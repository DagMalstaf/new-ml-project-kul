import pyspiel
import datetime
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from task4.agent.nn_mcts_agent import create_nn_mcts_agent
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard


def train_nn(game_name, num_simulations, exploration_coefficient, num_episodes, batch_size=32, validation_split=0.1):
    game = pyspiel.load_game(game_name)
    agent = create_nn_mcts_agent(game, num_simulations, exploration_coefficient)
    all_inputs = []
    all_policy_targets = []
    all_value_targets = []

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    for episode in range(num_episodes):
        print(f"Training episode %d out of %d episodes", episode, num_episodes)
        trajectory = run_self_play(agent, game)
        inputs, policy_targets, value_targets = process_trajectory(trajectory, agent.neural_network)
        all_inputs.extend(inputs)
        all_policy_targets.extend(policy_targets)
        all_value_targets.extend(value_targets)

        if (episode + 1) % batch_size == 0 or (episode + 1) == num_episodes:
            model_inputs = np.array(all_inputs)
            model_policy_targets = np.array(all_policy_targets)
            model_value_targets = np.array(all_value_targets)

            all_inputs, all_policy_targets, all_value_targets = [], [], []

            agent.neural_network.train_on_batch(model_inputs, [model_policy_targets, model_value_targets])

    callbacks = [
        ModelCheckpoint('task4/models/basic-nn-mcts-models/long-train-7x7-save.keras', save_best_only=True, monitor='loss'),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.0001, verbose=1),
        tensorboard_callback
    ]

    validation_data = (np.array(all_inputs), [np.array(all_policy_targets), np.array(all_value_targets)])
    agent.neural_network.fit(model_inputs, [model_policy_targets, model_value_targets],
                             epochs=10, callbacks=callbacks, validation_split=validation_split)

    agent.neural_network.save('task4/models/basic-nn-mcts-models/long-train-7x7.keras')


def process_trajectory(trajectory, model):
    if not trajectory:
        print("Empty trajectory received.")
        return np.array([]), [], [] 

    inputs = [np.array(state.observation_tensor()) for state, _ in trajectory]

    num_actions = model.policy_output.units  
    final_value = trajectory[-1][0].returns()[0]  
    policy_targets = [np.eye(num_actions)[action] for _, action in trajectory] 
    value_targets = [final_value] * len(trajectory) 

    if hasattr(inputs[0], 'shape'):
        inputs_shape = tuple(inputs[0].shape)
        inputs = np.array(inputs).reshape((-1,) + inputs_shape) 
    else:
        inputs = np.array(inputs)  

    return inputs, policy_targets, value_targets


def run_self_play(agent, game):
    state = game.new_initial_state()
    trajectory = []
    if state.is_terminal():
        print("Initial state is terminal. No actions possible.")
    while not state.is_terminal():
        action = agent.get_action(state)
        trajectory.append((state.clone(), action))
        state.apply_action(action)
    if not trajectory:
        print("No trajectory generated. Check game rules and initialization.")

    return trajectory
