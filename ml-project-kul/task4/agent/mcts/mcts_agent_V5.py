import numpy as np
import pyspiel
import logging
import hashlib
import time
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing


class Node:
    def __init__(self, state, game):
        self.state = state
        self.game = game
        self.children = {}
        self.visits = 0
        self.wins = 0

    def is_terminal(self):
        return self.state.is_terminal()

    def expand(self):
        legal_actions = self.state.legal_actions()
        for action in legal_actions:
            new_state = self.state.clone()
            new_state.apply_action(action)
            self.children[action] = Node(new_state, self.game)

    def select_child(self, exploration_coefficient):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            if child.visits > 0:
                uct_score = (child.wins / child.visits) + \
                            exploration_coefficient * np.sqrt(np.log(1 + self.visits) / (1 + child.visits))
            else:
                uct_score = float('inf')  # Encourage exploration of unvisited nodes

            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                best_child = child
        return best_action, best_child


class LocalMCTSAgent:
    def __init__(self, game, exploration_coefficient):
        self.game = game
        self.exploration_coefficient = exploration_coefficient
        self.root = None

    def hash_state(self, state):
        state_str = state.serialize()
        return hashlib.sha256(state_str.encode()).hexdigest()

    def run_simulation(self, initial_state):
        if not self.root:
            self.root = Node(initial_state, self.game)
        path = []
        current_node = self.root
    
        counter = 0
        while not current_node.is_terminal():
            if not current_node.children:
                current_node.expand()
            logging.info(f'Counter: {counter}')
            
            action, next_node = current_node.select_child(self.exploration_coefficient)
            counter += 1
            path.append((current_node, action, next_node))
            current_node = next_node
            current_node.visits += 1
        logging.info('Out of game simulation')
        result = result = current_node.state.returns()
        self.backpropagate(path, result)
        return self.root

    def backpropagate(self, path, result):
        for node, action, child in reversed(path):
            child.wins += result
            node.visits += 1

    def get_action(self, state, time_limit):
        self.run_simulation(state)
        _, best_child = self.root.select_child(self.exploration_coefficient)
        best_action = [action for action, child in self.root.children.items() if child == best_child][0]
        return best_action


    

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
    logging.info(f"MCTSAgent V5 created for tournament with player ID: {player_id}")
    num_simulations = 10
    exploration_coefficient = 1.414
    return LocalMCTSAgent(game, exploration_coefficient)


