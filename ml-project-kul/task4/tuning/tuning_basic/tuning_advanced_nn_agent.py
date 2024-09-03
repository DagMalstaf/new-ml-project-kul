import tensorflow as tf
import numpy as np
import optuna
import yaml
import os
import pyspiel
from tensorflow.keras.models import load_model, save_model

def load_parameters(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def optimize_hyperparameters():
    params = load_parameters('task4/tuning/parameters.yaml')
    strategy = get_distribution_strategy()
    print(f'Using strategy with devices: {strategy.num_replicas_in_sync}')
   
    def objective(trial):
        with strategy.scope(): 
            num_simulations = trial.suggest_int('num_simulations', params['num_simulations']['low'], params['num_simulations']['high'])
            exploration_coefficient = trial.suggest_float('exploration_coefficient', params['exploration_coefficient']['low'], params['exploration_coefficient']['high'])
            batch_size = trial.suggest_categorical('batch_size', params['batch_size']['options'])
            if strategy.num_replicas_in_sync > 1:
                batch_size *= strategy.num_replicas_in_sync
            epochs = trial.suggest_int('epochs', params['epochs']['low'], params['epochs']['high'])
            learning_rate = trial.suggest_float('learning_rate', params['learning_rate']['low'], params['learning_rate']['high'], log=True)
            optimizer_choice = trial.suggest_categorical('optimizer', params['optimizer']['options'])
            num_conv_layers = trial.suggest_int('num_conv_layers', params['num_conv_layers']['low'], params['num_conv_layers']['high'])
            num_dense_units = trial.suggest_categorical('num_dense_units', params['num_dense_units']['options'])
            dropout_rate = trial.suggest_uniform('dropout_rate', params['dropout_rate']['low'], params['dropout_rate']['high'])
            num_residual_blocks = trial.suggest_int('num_residual_blocks', params['num_residual_blocks']['low'], params['num_residual_blocks']['high'])

            game = pyspiel.load_game("dots_and_boxes(num_rows=3,num_cols=3)")
            agent = create_advanced_nn_mcts_agent(game, num_simulations, exploration_coefficient, num_conv_layers, num_dense_units, dropout_rate, num_residual_blocks, batch_size, epochs, learning_rate, optimizer_choice, model_path=None)
            
            trajectories = run_self_play(agent, game, num_episodes=10)
            inputs, policy_targets, value_targets = process_trajectory(trajectories)
            
            if len(inputs) % batch_size != 0:
                print('Warning: Last batch will be smaller than the others or dropped!')

            train_inputs, train_policy_targets, train_value_targets, val_inputs, val_policy_targets, val_value_targets = split_dataset(inputs, policy_targets, value_targets)
           
            if len(train_inputs) == 0 or len(val_inputs) == 0:
                raise ValueError("Training or validation dataset is empty. Check your data split or preprocessing.")

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

            history = agent.neural_network.fit(
                train_dataset,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_dataset,
                validation_steps=validation_steps
            )


            val_loss = history.history['val_loss'][-1]

            if trial.number == 0 or val_loss < trial.study.user_attrs.get("best_val_loss", float('inf')):
                trial.study.set_user_attr("best_val_loss", val_loss)
                save_model(agent.neural_network, 'task4/models/advanced-nn-mcts-models/best_tuned_model.keras')

        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=params['n_trials'])

    best_trial = study.best_trial
    results = {
        'best_trial_value': best_trial.value,
        'parameters': best_trial.params
    }

    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    with open('task4/tuning/results.yaml', 'w') as file:
        yaml.dump(results, file, default_flow_style=False)



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
            return tf.distribute.MirroredStrategy()
        else:
            return tf.distribute.get_strategy()
    except RuntimeError as e:
        print(e)
        return tf.distribute.get_strategy()

