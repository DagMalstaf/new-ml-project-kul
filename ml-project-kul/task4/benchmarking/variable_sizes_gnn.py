import os
import numpy as np
from task4.benchmarking.arena import Arena
from open_spiel.python.algorithms.mcts import MCTSBot
from open_spiel.python.bots.uniform_random import UniformRandomBot
import pyspiel
from task4.training.training_gnn.gnn_model import GNNetWrapper as gnn

from task4.training.training_gnn.gnn_evaluator import GNNEvaluator
from task4.training.training_gnn.gnn_agent import MCTSAgent
from tqdm import tqdm
import csv
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import logging

log = logging.getLogger(__name__)


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def benchmark_gnn(csv_path):
    current_directory = os.path.dirname(__file__)
    config_path = os.path.join(current_directory, 'config.yaml')
    config = load_config(config_path)

    with open(csv_path, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["size","MCTSWon","MCSTLost","MCTSDraw","MctsWinRate",
                            "RandomWon","RandomLost","RandomDraw","RandomWinRate"])

    numMCTSSimsTournament = config['numMCTSSimsTournament']

    for i in range(3, 15):
        log.warning(f'Running for size: {i}')
        num_rows = i
        num_cols = i

        game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
        game = pyspiel.load_game(game_string)

        nnet = gnn(config)
        nnet.load_checkpoint(current_directory, config['tournament_file_name'])

        evaluator = GNNEvaluator(nnet)

        gnn_agent = MCTSAgent(config['exploration_coefficient'], numMCTSSimsTournament, evaluator)

        random_agent = UniformRandomBot(1,np.random)

        arena = Arena(gnn_agent, random_agent, game)

        oneWonRandom, twoWonRandom, drawRandom = arena.playGamesAgainstRandom(gnn_agent, config['gamesToPlay'])
        
        RandomWinRate = round(oneWonRandom /config['gamesToPlay'], 2)

        oneWonMCTS, twoWonMCTS, drawMCTS = arena.playGamesAgainstMCTS(gnn_agent, config['gamesToPlay'])
        
        MCTSWinRate = round(oneWonMCTS /config['gamesToPlay'], 2)

        with open(csv_path, "a",newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([str(i),str(oneWonMCTS),str(twoWonMCTS),str(drawMCTS),str(MCTSWinRate),
                                str(oneWonRandom),str(twoWonRandom),str(drawRandom),str(RandomWinRate)])
            
        
        numMCTSSimsTournament += 5
    
def plot_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)

    if 'size' not in df.columns:
        print(f"'size' column not found in the DataFrame. Available columns: {df.columns}")
        return

    df['size'] = df['size'].astype(int)

    plt.figure(figsize=(10, 6))
    plt.plot(df['size'], df['MctsWinRate'], label='MCTS Win Rate', color='#4e79a7')
    plt.plot(df['size'], df['RandomWinRate'], label='Random Win Rate', color='#f28e2b')

    plt.xlabel('Board Size')
    plt.ylabel('Win Rate')
    plt.title('Win Rate vs Board Size')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

def main_benchmarking():
    current_directory = os.path.dirname(__file__)
    config_path = os.path.join(current_directory, 'config.yaml')
    config = load_config(config_path)
    csv_file_name = config['resultsFilePathBench']
    csv_path = os.path.join(current_directory, csv_file_name)

    #benchmark_gnn(csv_path)
    plot_results(csv_path)

if __name__ == "__main__":
    main_benchmarking()
    