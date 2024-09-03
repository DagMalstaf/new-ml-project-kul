import numpy as np
import pyspiel
import logging
import hashlib
import time
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing


class LocalMCTSAgent:
    def __init__(self, game, exploration_coefficient):
        self.local_tree = {}
        self.local_simulations_count = 0
        self.game = game
        self.exploration_coefficient = exploration_coefficient

    def hash_state(self, state):
        state_str = str(state)
        return hashlib.sha256(state_str.encode()).hexdigest()

    def run_simulation(self, state):
        self.local_simulations_count += 1
        path = []
        while not state.is_terminal():
            state_key = self.hash_state(state)
            if state_key not in self.local_tree:
                self.local_tree[state_key] = self.populate_tree_with_new_state(state)

            action = self.select_action(state_key, self.local_tree)
            
            if isinstance(action, (int, np.int64)):
                state.apply_action(action)
                path.append((state_key, action))
            else:
                raise Exception('Not integer error')
            break
        logging.info(f"Rewards: {state.rewards()}")

        for state_key, action in reversed(path):
            action_idx = self.local_tree[state_key]['action_to_index'][action]
            self.local_tree[state_key]['N'][action_idx] += 1
            self.local_tree[state_key]['W'][action_idx] += state.returns()[0]
            self.local_tree[state_key]['Q'][action_idx] = self.local_tree[state_key]['W'][action_idx] / self.local_tree[state_key]['N'][action_idx]
            
        return self.local_tree, self.local_simulations_count
        

    def populate_tree_with_new_state(self, state):
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)
        action_to_index = {action: idx for idx, action in enumerate(legal_actions)}
        return {
            'Q': np.zeros(num_actions),
            'N': np.zeros(num_actions, dtype=int),
            'W': np.zeros(num_actions),
            'legal_actions': np.array(legal_actions),
            'action_to_index': action_to_index
        }
            
    def select_action(self, state_key, tree):
        node = tree[state_key]
        total_visits = np.sum(node['N'])
        logging.info(f"Node N: {node['N']}")
        logging.info(f"Total visits: {total_visits}")
        U = self.exploration_coefficient * np.sqrt(np.log(total_visits) / (node['N']))
        values = node['Q'] + np.where(node['N'] > 0, U, float('inf'))
        best_action_index = np.argmax(values)
        logging.info(f"Best action: {best_action_index}")
        return node['legal_actions'][best_action_index]
    '''
     def select_action(self, state_key, tree):
        node = tree[state_key]
        n_i = np.sum(node['N'])
        w_i = np.sum(node['W'])
        N_i = 

        exploitation = n_i / w_i
        exploration = self.exploration_coefficient * np.sqrt(np.log(N_i) / (n_i))
        UCT = exploitation + exploration

        return node['legal_actions'][best_action_index]
    '''
   
    

class MCTSAgent:
    def __init__(self, game, num_simulations, exploration_coefficient):
        self.game = game
        self.num_simulations = num_simulations
        self.exploration_coefficient = exploration_coefficient
        self.global_tree = {}
        self.simulations_count = 0
        self.process_pool = Pool(processes=4)
    
    def __del__(self):
        self.close_pool()

    def get_action(self, state, time_limit):
        start_time = time.time()
        while time.time() - start_time < time_limit: 
            agents = [LocalMCTSAgent(self.game, self.exploration_coefficient) for _ in range(self.process_pool.ncpus)]
            cloned_states = [state.clone() for _ in range(self.process_pool.ncpus)]
            result_objs = self.process_pool.amap(lambda ag, st: ag.run_simulation(st), agents, cloned_states)
            result_objs.wait()

        for local_tree, local_simulations_count in result_objs.get():
            self.simulations_count += local_simulations_count
            self.aggregate_results(local_tree)

        root_state_key = self.hash_state(state)
        best_action_idx = np.argmax(self.global_tree[root_state_key]['N'])
        return self.global_tree[root_state_key]['legal_actions'][best_action_idx]
    
    def hash_state(self, state):
        state_str = str(state)
        return hashlib.sha256(state_str.encode()).hexdigest()

    def aggregate_results(self, local_tree):
        for key, data in local_tree.items():
            if key not in self.global_tree:
                self.global_tree[key] = data.copy()
            else:
                self.global_tree[key]['N'] += data['N']
                self.global_tree[key]['W'] += data['W']
                non_zero = self.global_tree[key]['N'] != 0
                self.global_tree[key]['Q'][non_zero] = self.global_tree[key]['W'][non_zero] / self.global_tree[key]['N'][non_zero]
                self.global_tree[key]['Q'][~non_zero] = 0


    def close_pool(self):
        if self.process_pool:
            self.process_pool.close()
            self.process_pool.join()
            self.process_pool = None


def get_agent_for_tournament(player_id):
    game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})
    logging.info(f"MCTSAgent V3 created for tournament with player ID: {player_id}")
    num_simulations = 10
    exploration_coefficient = 1.41
    return MCTSAgent(game, num_simulations, exploration_coefficient)


