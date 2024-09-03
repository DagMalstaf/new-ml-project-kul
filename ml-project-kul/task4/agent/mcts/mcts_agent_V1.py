import numpy as np
import random
import pyspiel
import os
import logging
import hashlib
import time

class MCTSAgent:
    def __init__(self, game, num_simulations, exploration_coefficient):
        self.game = game
        self.num_simulations = num_simulations
        self.exploration_coefficient = exploration_coefficient
        self.tree = {}
        self.simulations_count = 0

    def hash_state(self, state):
        state_str = str(state) 
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def run_simulation(self, state):
        self.simulations_count += 1
        path = []
        while not state.is_terminal():
            state_key = self.hash_state(state)
            if state_key not in self.tree:
                self.populate_tree_with_new_state(state, state_key)

            action = self.select_action(state_key)
            if isinstance(action, int):
                state.apply_action(action)
                path.append((state_key, action))
                logging.info(action)
            else:
                raise Exception('Not integer error')

        if path:
            reward = state.returns()[0]  
            for state_key, action in reversed(path):
                self.tree[state_key]['N'][action] += 1
                self.tree[state_key]['W'][action] += reward
                self.tree[state_key]['Q'][action] = self.tree[state_key]['W'][action] / self.tree[state_key]['N'][action]


    def populate_tree_with_new_state(self, state, state_key):
        legal_actions = state.legal_actions()
        self.tree[state_key] = {
            'Q': {a: 0.0 for a in legal_actions},
            'N': {a: 0 for a in legal_actions},
            'W': {a: 0.0 for a in legal_actions},
            'legal_actions': legal_actions
        }
        
    def select_action(self, state_key):
        best_action = None
        best_value = float('-inf')
        node = self.tree[state_key]
        total_visits = sum(node['N'].values()) + 1  

        for action in node['legal_actions']:
            Q = node['Q'][action]
            N = node['N'][action]
            U = self.exploration_coefficient * np.sqrt(np.log(total_visits) / (1 + N)) if N > 0 else float('inf')
            value = Q + U
            if value > best_value:
                best_value = value
                best_action = action

        return best_action


    def get_action(self, state, time_limit):
        self.simulations_count = 0
        start_time = time.time()
        while time.time() - start_time < time_limit: 
            self.run_simulation(state.clone())

        state_key = self.hash_state(state)
        most_visited_action = max(self.tree[state_key]['N'], key=self.tree[state_key]['N'].get)
        return most_visited_action
    
def get_agent_for_tournament(player_id):
    game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})
    logging.info(f"MCTSAgent V1 created for tournament with player ID: {player_id}")
    num_simulations = 10
    exploration_coefficient = 1.41
    return MCTSAgent(game, num_simulations, exploration_coefficient)


