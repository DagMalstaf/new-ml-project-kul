# encoding: utf-8
"""
tournament.py

Code to play a round robin tournament between dots-and-boxes agents.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""
import importlib.util
import itertools
import logging
import os
import sys
from pathlib import Path
import errno
import os
import signal
import functools

import click
import pandas as pd
import numpy as np
from tqdm import tqdm

import pyspiel
from open_spiel.python.algorithms.evaluate_bots import evaluate_bots

log = logging.getLogger(__name__)

def load_agent(path, player_id):
    """Initialize an agent from a 'dotsandboxes_agent.py' file.
    """
    module_dir = os.path.dirname(os.path.abspath(path))
    sys.path.insert(1, module_dir)
    spec = importlib.util.spec_from_file_location("dotsandboxes_agent", path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo.get_agent_for_tournament(player_id)


def load_agent_from_dir(agent_id, path):
    """Scrapes a directory for an dots-and-boxes agent.

    This function searches all subdirectories for an 'dotsandboxes_agent.py' file and
    calls the ``get_agent_for_tournament`` method to create an instance of
    a player 1 and player 2 agent. If multiple matching files are found,
    a random one will be used.
    """
    agent_path = next(Path(path).glob('**/dotsandboxes_agent.py'))
    log.info(agent_path)
    try:
        return {
            'id':  agent_id,
            'agent_p1': load_agent(agent_path, 0),
            'agent_p2': load_agent(agent_path, 1),
        }
    except Exception as e:
        log.exception("Failed to load %s" % agent_id)

class TimeoutError(Exception):
    pass

def timeout(seconds=1., error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, seconds) 
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def play_match(game, agent1, agent2, seed=1234, rounds=100, timeout_ms=200):
    """Play a set of games between two agents.
    """
    # Set timers for each agent
    for agent in (agent1, agent2):
        for player in ('agent_p1', 'agent_p2'):
            agent[player].step = timeout(seconds=timeout_ms/1000, error_message=f"Timeout for agent {agent['id']}")(agent[player].step)
    # Play tournament
    rng = np.random.RandomState(seed)
    results = []
    for i in tqdm(range(rounds)):
        # Alternate between the two agents as p1 and p2
        for (p1, p2) in [(agent1, agent2), (agent2, agent1)]:
            try:
                returns = evaluate_bots(
                        game.new_initial_state(),
                        [p1['agent_p1'], p2['agent_p2']],
                        rng)
                error = None
            except Exception as ex:
                log.exception("Failed to play between %s and %s" % (agent1['id'], agent2['id']))
                template = "An exception of type {0} occurred. Message: {1}"
                error = template.format(type(ex).__name__, ex)
                returns = [None, None]
            finally:
                results.append({
                    "agent_p1": p1['id'],
                    "agent_p2": p2['id'],
                    "return_p1": returns[0],
                    "return_p2": returns[1],
                    "error": error
                })
            log.info(f"Game {i}: %s" % results[i])
    return results


def process_results(results_df):
    if results_df.empty:
        log.error("Results DataFrame is empty. No matches to process.")
        return
    win_counts = {}

    for _, row in results_df.iterrows():
        if row['error']:
            log.error(f"Error in match: {row['error']}")
            continue

        for agent in [row['agent_p1'], row['agent_p2']]:
            if agent not in win_counts:
                win_counts[agent] = {'wins': 0, 'losses': 0, 'draws': 0}

        if row['return_p1'] > row['return_p2']:
            win_counts[row['agent_p1']]['wins'] += 1
            win_counts[row['agent_p2']]['losses'] += 1
        elif row['return_p1'] < row['return_p2']:
            win_counts[row['agent_p1']]['losses'] += 1
            win_counts[row['agent_p2']]['wins'] += 1
        else:
            win_counts[row['agent_p1']]['draws'] += 1
            win_counts[row['agent_p2']]['draws'] += 1

    for agent, counts in win_counts.items():
        log.info(f"Agent {agent} - Wins: {counts['wins']}, Losses: {counts['losses']}, Draws: {counts['draws']}")

    return win_counts

    
@click.command()
@click.argument('agent1_id', type=str)
@click.argument('agent1_dir', type=click.Path(exists=True))
@click.argument('agent2_id', type=str)
@click.argument('agent2_dir', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=False))
@click.option('--rounds', default=20, help='Number of rounds to play.')
@click.option('--num_rows', default=5, help='Number of rows.')
@click.option('--num_cols', default=5, help='Number of cols.')
@click.option('--timeout', default=200000000, help='Max time (in ms) to reply')
@click.option('--seed', default=1234, help='Random seed')
def cli(agent1_id, agent1_dir, agent2_id, agent2_dir, output, rounds, timeout, num_rows, num_cols, seed):
    """Play a set of games between two agents."""
    logging.basicConfig(level=logging.INFO)
    logging.info('KUL TOURNAMENT')
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows={},"
        "num_cols={})").format(num_rows, num_cols)
    log.info("Creating game: {}".format(dotsandboxes_game_string))
    game = pyspiel.load_game(dotsandboxes_game_string)

    # Load the agents
    log.info("Loading the agents")
    agent1 = load_agent_from_dir(agent1_id, agent1_dir)
    agent2 = load_agent_from_dir(agent2_id, agent2_dir)
    log.info("Loaded agents: {}".format(agent1))
    log.info("Loaded agents: {}".format(agent2))

    # Play the tournament
    log.info("Playing {} matches between {} and {}".format(rounds, agent1_id,  agent2_id))
    results = play_match(game, agent1, agent2, seed, rounds, timeout)
    
    # Process the results
    log.info("Processing the results")
    results = pd.DataFrame(results)

    # Save and process the results
    results.to_csv(output, index=False)
    log.info("Results saved to {}".format(output))

    # Process results to log the number of wins
    win_counts = process_results(results)
    log.warning(win_counts)
    #log.warning(f"Average time/step by agent: {agent1['agent_p1']}: {agent1['agent_p1'].get_average_time():.4f}")


if __name__ == '__main__':
    sys.exit(cli())
 