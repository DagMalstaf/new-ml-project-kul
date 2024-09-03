#!/usr/bin/env python3

import yaml
from task4.play_tournament.new_kul_tournament import cli
from click.testing import CliRunner
import random
import logging
from datetime import datetime
from colorlog import ColoredFormatter
from task4.tuning.tuning_basic.tuning_advanced_nn_agent import optimize_hyperparameters
from task4.training.training_alphazero.main_training import main_training
from task4.training.training_gnn.main_training_gnn import main_training_gnn
from task4.benchmarking.variable_sizes_gnn import main_benchmarking
from task4.tuning.tuning_gnn.tuning_gnn import main_tuning

log = logging.getLogger(__name__)

def load_config(path):
    """ Load YAML configuration file. """
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def call_tournament(num_rows, num_cols, agent, rounds):
    if not isinstance(num_rows, int) or not isinstance(num_cols, int) or not isinstance(rounds, int):
        raise ValueError("num_rows, num_cols, and rounds must be integers")
    if num_rows <= 0 or num_cols <= 0 or rounds <= 0:
        raise ValueError("num_rows, num_cols, and rounds must be positive integers")
    
    runner = CliRunner()
    result = runner.invoke(cli, [str(agent), f'submit_files/{str(agent)}/', 'MCTS', 'task4/agent/mcts_openspiel/', 'task4/agent/results.txt', '--rounds', str(rounds), '--num_rows', str(num_rows), '--num_cols', str(num_cols)])
    if result.exit_code == 0:
        log.info("Tournament ran successfully.")
        log.info(result.output)
    else:
        log.warning("Error running tournament:")
        log.warning(result.exception)
        log.warning(result.output)
    return

def setup_logging():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f'training_logs_{current_time}.log'

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    fh = logging.FileHandler(log_file_name, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(console_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

def main(config_path):
    setup_logging()
    log = logging.getLogger(__name__)
    try:
        log.info("Starting process...")
        while True:
            log.info("Choose an action:")
            print("1. Run the benchmarks")
            print("2. Tune the GNN model")
            print("3. Train the GNN model")
            print("4. Train the AlphaZero model")
            print("5. Run a tournament against a random agent")
            print("6. Exit")
            choice = input("Enter the number of your choice: ")

            if choice == '1':
                main_benchmarking()
            elif choice == '2':
                main_tuning()
            elif choice == '3':
                main_training_gnn()
            elif choice == '4':
                main_training()
            elif choice == '5':
                sub_choice = input("Do you want to input the number of rows and columns yourself? (y/n): ")
                if sub_choice.lower() == 'y':
                    num_rows = int(input("Enter number of rows: "))
                    num_cols = int(input("Enter number of columns: "))
                else:
                    num_rows = random.randint(3, 20)
                    num_cols = random.randint(3, 20)
                
                rounds = int(input("Enter number of rounds: "))
                log.info("Choose an agent:")
                print("1. alphazero")
                print("2. gnn_var")
                print("3. mcts")
                print("4. tournament")
                agent_choice = input("Enter the number of your choice: ")
                if agent_choice == '1':
                    agent = 'alphazero'
                elif agent_choice == '2':
                    agent = 'gnn_var'
                elif agent_choice == '3':
                    agent = 'mcts'
                elif agent_choice == '4':
                    agent = 'tournament'
                else:
                    log.warning("Invalid choice. Defaulting to 'mcts'.")
                    agent = 'mcts'
                
                call_tournament(num_rows, num_cols, agent, rounds)
            elif choice == '6':
                log.warning("Exiting.")
                break
            else:
                log.warning("Invalid choice. Please try again.")
    except Exception as e:
        log.exception("An error occurred during the process")
        raise
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'task4/training/config.yaml'
    main(config_path)
