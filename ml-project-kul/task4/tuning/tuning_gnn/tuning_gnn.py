import os
import yaml
import optuna
import torch
from optuna.trial import TrialState
import pyspiel

from task4.tuning.tuning_gnn.coach import Coach
from task4.tuning.tuning_gnn.gnn_model import GNNetWrapper 
from task4.training.training_gnn.main_training_gnn import main_training_gnn

def save_updated_config(best_params, config_path):
    config = load_config(config_path)
    
    config['num_channels'] = best_params['num_channels']
    config['lr'] = best_params['lr']
    config['l2_coeff'] = best_params['l2_coeff']
    config['epocs_gnn'] = best_params['epocs_gnn']
    config['batch_size'] = best_params['batch_size']
    config['exploration_coefficient'] = best_params['exploration_coefficient']

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def objective(trial):
    config = {
        'num_channels': trial.suggest_int('num_channels', 64, 256),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'l2_coeff': trial.suggest_float('l2_coeff', 1e-5, 1e-2, log=True),
        'epocs_gnn': trial.suggest_int('epocs_gnn', 10, 50),
        'batch_size': trial.suggest_int('batch_size', 32, 128),
        'num_MCTS_sims': 15,
        'exploration_coefficient': trial.suggest_float('exploration_coefficient', 0.1, 2.0, log=True),
        'tempThreshold': 15,
        'maxlenOfQueue': 200000,
        'numIterations': 1,
        'numEpisodes': 10,
        'updateThreshold': 0.55,
        'arenaCompare': 10,
        'numItersForTrainExamplesHistory': 10000,
        'checkpoint': './checkpoints',
        'saveResults': False,
        'resultsFilePath': 'results.csv',
        'load_folder_file_0': "checkpoint_last_attempt",
        'load_folder_file_1': "checkpoint_13",
        'tune_file_name': 'best.weights.h5'
    }
    current_directory = os.path.dirname(__file__)
    game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})
    nnet = GNNetWrapper(config)
    coach = Coach(game, nnet, config)

    coach.loadTrainExamples()

    winning_rate = coach.learn()

    if winning_rate > 0.70:
        trial.study.stop()
        best_params = trial.params
        current_directory = os.path.dirname(__file__)
        config_path = os.path.join(current_directory, 'config.yaml')
        save_updated_config(best_params, config_path)
        config = load_config(config_path)
        main_training_gnn(config)

    return winning_rate

def main_tuning():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000, timeout=21000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_params = trial.params
    with open("best_hyperparameters.yaml", "w") as f:
        yaml.dump(best_params, f)

if __name__ == "__main__":
    main_tuning()