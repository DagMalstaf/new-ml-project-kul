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
            
            if isinstance(action, (int, np.int64)):
                state.apply_action(action)
                path.append((state_key, action))
            else:
                raise Exception('Not integer error')
            
        if path:
            reward = state.returns()[0]
            for state_key, action in reversed(path):
                action_idx = self.tree[state_key]['action_to_index'][action]
                self.tree[state_key]['N'][action_idx] += 1
                self.tree[state_key]['W'][action_idx] += reward
                self.tree[state_key]['Q'][action_idx] = self.tree[state_key]['W'][action_idx] / self.tree[state_key]['N'][action_idx]

    def populate_tree_with_new_state(self, state, state_key):
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)
        action_to_index = {action: idx for idx, action in enumerate(legal_actions)}
        self.tree[state_key] = {
            'Q': np.zeros(num_actions),
            'N': np.zeros(num_actions, dtype=int),
            'W': np.zeros(num_actions),
            'legal_actions': np.array(legal_actions),
            'action_to_index': action_to_index
        }
            
    def select_action(self, state_key):
        node = self.tree[state_key]
        total_visits = np.sum(node['N']) + 1
        U = self.exploration_coefficient * np.sqrt(np.log(total_visits) / (1 + node['N']))
        values = node['Q'] + np.where(node['N'] > 0, U, float('inf'))
        best_action_index = np.argmax(values)
        if best_action_index >= len(node['legal_actions']):
            raise IndexError("Best action index out of bounds")
        return node['legal_actions'][best_action_index]

    
    def get_action(self, state, time_limit):
        self.simulations_count = 0
        start_time = time.time()
        while time.time() - start_time < time_limit: 
            self.run_simulation(state.clone())
        state_key = self.hash_state(state)
        most_visited_action_index = np.argmax(self.tree[state_key]['N'])
        return self.tree[state_key]['legal_actions'][most_visited_action_index]


def get_agent_for_tournament(player_id):
    game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})
    logging.info(f"MCTSAgent V2 created for tournament with player ID: {player_id}")
    num_simulations = 10
    exploration_coefficient = 1.41
    return MCTSAgent(game, num_simulations, exploration_coefficient)


