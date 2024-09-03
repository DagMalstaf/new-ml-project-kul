import numpy as np
import random
import pyspiel
import os
import logging
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def build_cython_module():
    setup_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mcts')
    setup_file = os.path.join(setup_path, "setup.py")
    build_command = [sys.executable, setup_file, 'build_ext', '--inplace']

    current_dir = os.getcwd()
    os.chdir(setup_path)

    try:
        subprocess.check_call(build_command)
    except subprocess.CalledProcessError as e:
        logging.error("Failed to build Cython module: %s", e)
        raise RuntimeError("Cython module build failed")
    finally:
        os.chdir(current_dir)

build_cython_module()

from task4.agent.mcts.mcts_simulation_cython import MCTSAgent as CythonMCTSAgent

class MCTSAgent:
    def __init__(self, game, num_simulations, exploration_coefficient):
        self.game = game
        self.num_simulations = num_simulations
        self.exploration_coefficient = exploration_coefficient
        self.agent = CythonMCTSAgent(game.num_distinct_actions(), self.exploration_coefficient)

    def serialize_state(self, state):
        """ Returns a string serialization of the state for dictionary keys. """
        return str(state)

    def run_simulation(self, state):
        logging.info('--------------------------------------------------------')
        while not state.is_terminal():
            legal_actions = state.legal_actions()
            self.agent.update_legal_actions(legal_actions)
            
            action, value = self.agent.get_action()
            logging.info(f"Given value for action: {value}")
            state.apply_action(action)

            updated_legal_actions = state.legal_actions()
            self.agent.update_legal_actions(updated_legal_actions)

        reward = state.returns()[0]
        logging.info(f'Final reward: {reward}')
        self.agent.update_tree(reward)
        logging.info('--------------------------------------------------------')

    def get_action(self, state):
        """ Determine the best action by running multiple simulations and choosing the most visited one. """
        for i in range(self.num_simulations):
            logging.info(f'Running simulation: {i}')
            cloned_state = state.clone()
            self.run_simulation(cloned_state)
        return self.agent.get_action() 

def get_agent_for_tournament(player_id):
    logging.info("Building Cython module for MCTS agent...")
    build_cython_module()

    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})
    logging.info(f"MCTSAgent created for tournament with player ID: {player_id}")
    
    num_simulations = 2
    exploration_coefficient = 1.41

    return MCTSAgent(game, num_simulations, exploration_coefficient)
