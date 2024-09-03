#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import pyspiel
import random
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator, MCTSBot
from open_spiel.python.algorithms import evaluate_bots
import sys
import os
import time
import signal
import functools
import errno


import yaml
from gnn_evaluator import GNNEvaluator
from gnn_model import GNNetWrapper as gnn


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

class Agent(pyspiel.Bot):
    def __init__(self, player_id, exploration, sims, evaluator):
        pyspiel.Bot.__init__(self)
        self.player_id = player_id
        self.game = None
        self.evaluator = evaluator
        self.exploration = exploration
        self.sims = sims
        self.bot = None
        self.player_id = player_id
       
    def restart_at(self, state):
        pass

    def inform_action(self, state, player_id, action):
        pass

    #@timeout(seconds=0.180, error_message="Timeout for prediction")
    def predict(self, state):
        start_time = time.time()
        result = self.bot.step(state)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Predict function took {elapsed_time:.6f} seconds")
        return result
    
    #@timeout(seconds=0.050, error_message="Timeout for prediction")
    def fallback(self, state):
        start_time = time.time()
        try:
            result = self.evaluator.prior(state)
            best_action = self.evaluator.get_best_action(result)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Fallback function took {elapsed_time:.6f} seconds")
            return best_action
        except TimeoutError:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Fallback function timed out after {elapsed_time:.6f} seconds")
            #print("Fallback Timed out")
            return self.random_fallback(state)

    def random_fallback(self, state):
        start_time = time.time()
        legal_actions = state.legal_actions()
        result = random.choice(legal_actions)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Random fallback function took {elapsed_time:.6f} seconds")
        return result

    def step(self, state):
        if self.game is None:
            self.game = state.get_game()
            self.bot = MCTSBot(self.game, self.exploration, self.sims, self.evaluator)
        try:
            result = self.predict(state)
            return result
        except TimeoutError:
            #print("Prediction timed out")
            action = self.fallback(state)
        return action

def get_agent_for_tournament(player_id):
    current_directory = os.path.dirname(__file__)
    config_path = os.path.join(current_directory, 'config.yaml')
    config = load_config(config_path)

    nnet = gnn(config)
    
    nnet.load_checkpoint(current_directory, config['tournament_file_name'])
    
    evaluator = GNNEvaluator(nnet)

    agent = Agent(player_id, config['exploration_coefficient'], config['numMCTSSimsTournament'], evaluator)

    print(f"GNN Agent created for tournament with player ID: {player_id}")
    return agent

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=7,num_cols=7)")
    game = pyspiel.load_game(dotsandboxes_game_string)
    bots = [get_agent_for_tournament(0), MCTSBot(game,1.14,2, RandomRolloutEvaluator(5))]
    for _ in range(1):
        returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
        print(returns[0])
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())