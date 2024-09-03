import yaml
from task4.training.training_gnn.coach import Coach
from task4.training.training_gnn.gnn_model import GNNetWrapper as gnn
import pyspiel
import logging
from task4.training.training_alphazero.resnet_model_V0 import get_distribution_strategy
import torch 
from task4.training.training_gnn.greedy_agent import *
from task4.training.training_gnn.minimax_agent import *


log = logging.getLogger(__name__)

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def main_training_gnn(config=None):
    if config is None:
        config_path = 'task4/training/training_gnn/config.yaml'
        config = load_config(config_path)
    num_rows = config['num_rows']
    num_cols = config['num_cols']
    game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
    game = pyspiel.load_game(game_string)

    strategy = get_distribution_strategy()  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.warning(f'Training on {"GPU" if device.type == "cuda" else "CPU"}')

    with strategy.scope():
        neural_net = gnn(config).to(device) 
        
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
    main_training_gnn()