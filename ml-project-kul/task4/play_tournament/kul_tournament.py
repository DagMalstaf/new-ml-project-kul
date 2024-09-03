import sys
sys.path.append('/Users/dagmalstaf/Documents/GITHUB/ml-project-kul')

import itertools
import logging
import importlib.util
import sys
import os
import click
import pyspiel

import numpy as np

from tqdm import tqdm
from pathlib import Path
from open_spiel.python.algorithms.evaluate_bots import evaluate_bots
import time


BLUE = "\033[1;34m"
RESET = "\033[0m"

logging.basicConfig(level=logging.INFO, format=f'{BLUE}%(asctime)s - %(levelname)s - %(message)s{RESET}')

def load_agent(path, player_id):
    module_dir = os.path.dirname(os.path.abspath(path))
    sys.path.insert(1, module_dir)
    agent_module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(agent_module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_agent_for_tournament(player_id)


def play_game(game, agent1, agent2, time_limit):
    state = game.new_initial_state()
    metrics = {
        agent1: {'total_simulations': 0, 'move_times': [], 'actions_taken': 0},
        agent2: {'total_simulations': 0, 'move_times': [], 'actions_taken': 0}
    }

    while not state.is_terminal():
        current_player = state.current_player()
        if current_player == -4: 
            logging.info("Game has reached a terminal state.")
            break

        current_agent = agent1 if current_player == 0 else agent2
        start_time = time.time()
        action = current_agent.get_action(state, time_limit)
        end_time = time.time()

        if action not in state.legal_actions():
            logging.error("Action is not legal")
            continue

        state.apply_action(action)
        metrics[current_agent]['move_times'].append(end_time - start_time)
        metrics[current_agent]['total_simulations'] += current_agent.simulations_count
        metrics[current_agent]['actions_taken'] += 1

    return state.returns(), metrics


def play_match(game, agent1, agent2, rounds, time_limit):
    results = []
    aggregated_metrics = {agent1: {'total_simulations': 0, 'move_times': [], 'actions_taken': 0},
                          agent2: {'total_simulations': 0, 'move_times': [], 'actions_taken': 0}}

    for i in tqdm(range(rounds), desc="Match progress"):
        logging.info(f"Playing round: {i} out of rounds: {rounds}")
        scores1, game_metrics1 = play_game(game, agent1, agent2, time_limit)
        update_metrics(aggregated_metrics, game_metrics1)
        results.append(scores1)
        
        scores2, game_metrics2 = play_game(game, agent2, agent1, time_limit)
        scores2_reversed = [scores2[1], scores2[0]]
        update_metrics(aggregated_metrics, game_metrics2)
        results.append(scores2_reversed)
            
    averages = {agent: {
        'average_time_per_action': np.mean(metrics['move_times']) if metrics['move_times'] else 0,
        'average_simulations_per_action': metrics['total_simulations'] / metrics['actions_taken'] if metrics['actions_taken'] else 0,
        'total_actions': metrics['actions_taken'],
        'total_simulations': metrics['total_simulations']
    } for agent, metrics in aggregated_metrics.items()}

    return np.array(results), aggregated_metrics, averages

def update_metrics(aggregated_metrics, game_metrics):
    for agent, metrics in game_metrics.items():
        aggregated_metrics[agent]['total_simulations'] += metrics['total_simulations']
        aggregated_metrics[agent]['move_times'].extend(metrics['move_times'])
        aggregated_metrics[agent]['actions_taken'] += metrics['actions_taken']


def log_agent_stats(agent, data):
    logging.info(f"{agent}:")
    logging.info(f"  Average Time per Action: {data['average_time_per_action']:.4f}s")
    logging.info(f"  Average Simulations per Action: {data['average_simulations_per_action']:.2f}")
    logging.info(f"  Total Actions Taken: {data['total_actions']}")
    logging.info(f"  Total Simulations: {data['total_simulations']}")

def log_performance_comparison(agent1, agent2, averages):
    if averages[agent1]['average_simulations_per_action'] != 0:
        sim_performance_difference = ((averages[agent2]['average_simulations_per_action'] - averages[agent1]['average_simulations_per_action']) / averages[agent1]['average_simulations_per_action']) * 100
        logging.info(f"Comparison:")
        logging.info(f"  Agent 2 is {'more' if sim_performance_difference > 0 else 'less'} efficient by {abs(sim_performance_difference):.2f}% in terms of average simulations per action compared to Agent 1.")
    else:
        logging.info("Comparison:")
        logging.info("  Unable to calculate simulation efficiency difference: Agent 1 conducted no simulations.")


@click.command()
@click.argument('agent1_dir', type=click.Path(exists=True))
@click.argument('agent2_dir', type=click.Path(exists=True))
@click.option('--rounds', default=20, help='Number of rounds to play.')
@click.option('--num_rows', default=2, help='Number of rows.')
@click.option('--num_cols', default=2, help='Number of columns.')
@click.option('--seed', default=1234, help='Random seed.')
def cli(agent1_dir, agent2_dir, rounds, num_rows, num_cols, seed):
    np.random.seed(seed)
    game = pyspiel.load_game(f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols})")
    agent1_path = next(Path(agent1_dir).glob('**/mcts_os_agent_V1.py'))
    agent2_path = next(Path(agent2_dir).glob('**/mcts_os_agent_V4.py'))
    agent1 = load_agent(str(agent1_path), player_id=0)
    agent2 = load_agent(str(agent2_path), player_id=1)
    time_limit = 0.001
    logging.info("Starting the tournament...")

    results, metrics, averages = play_match(game, agent1, agent2, rounds, time_limit)


    wins = [np.sum(results[:, 0] == 1), np.sum(results[:, 1] == 1)]
    losses = [np.sum(results[:, 0] == -1), np.sum(results[:, 1] == -1)]
    draws = results.shape[0] * 2 - (wins[0] + wins[1] + losses[0] + losses[1])
    logging.info("================================================================")
    logging.info("Tournament Results:")
    logging.info(f"  Agent1 Wins: {wins[0]}, Agent2 Wins: {wins[1]}, Draws: {draws}")

    log_agent_stats("Agent1", averages[agent1])
    log_agent_stats("Agent2", averages[agent2])
    log_performance_comparison(agent1, agent2, averages)


if __name__ == '__main__':
    sys.exit(cli())

