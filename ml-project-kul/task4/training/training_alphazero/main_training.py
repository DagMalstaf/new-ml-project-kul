import yaml
from task4.training.training_alphazero.coach import Coach
from task4.training.training_alphazero.resnet_model_V0 import NNetWrapper as nn
import pyspiel
import logging
from task4.training.training_alphazero.resnet_model_V0 import get_distribution_strategy

log = logging.getLogger(__name__)

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main_training():
    config_path = 'task4/training/training_alphazero/config.yaml'
    config = load_config(config_path)
    num_rows = config['num_rows']
    num_cols = config['num_cols']
    game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
    game = pyspiel.load_game(game_string)

    strategy = get_distribution_strategy()  

    with strategy.scope():
        neural_net = nn(game,
                        config['checkpoint_path'], 
                        config['learning_rate'],
                        config['nn_width'],
                        config['nn_depth'],
                        config['weight_decay'],
                        config)
        
        if config['load_model']:
            log.info('Loading model')
            neural_net.load_checkpoint(config['checkpoint'], 'checkpoint_11.weights.h5')
        else:
            log.warning('Not loading a checkpoint!')
        
        coach = Coach(game, neural_net, config)

        if config['load_examples']:
            coach.loadTrainExamples()
        coach.learn()

if __name__ == "__main__":
    main_training()