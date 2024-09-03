import numpy as np
import pyspiel
import logging
import hashlib
import time
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing
import open_spiel.python.algorithms.mcts as mcts
from task4.agent.mcts_openspiel.evaluator_V1 import RandomRolloutEvaluator

class LocalMCTSAgent(mcts.MCTSBot):
    def __init__(self, game, uct_c, max_simulations, evaluator, verbose=False):
        super().__init__(game, uct_c, max_simulations, evaluator)
        self.verbose = verbose
        self.simulations_count = 0

    def get_action(self, state, time):
        root = self.mcts_search(state)
        best = root.best_child()
        mcts_action = best.action
        self.simulations_count += 10
        return mcts_action

def get_agent_for_tournament(player_id):
    game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})
    logging.info(f"MCTSAgent OS V0 created for tournament with player ID: {player_id}")
    num_simulations = 10
    exploration_coefficient = 1.414
    evaluator = RandomRolloutEvaluator(5)
    return LocalMCTSAgent(game, exploration_coefficient, num_simulations, evaluator)


