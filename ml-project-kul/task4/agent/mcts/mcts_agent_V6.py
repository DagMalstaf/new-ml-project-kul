import numpy as np
import pyspiel
import logging
import hashlib
import time
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing
import open_spiel.python.algorithms.mcts as mcts
from open_spiel.python.algorithms.mcts import Evaluator

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


class LocalMCTSAgent(mcts.MCTSBot):
    def __init__(self, game, uct_c, max_simulations, evaluator, verbose=False):
        super().__init__(game, uct_c, max_simulations, evaluator)
        self.verbose = verbose
        self.simulations_count = 0

    def get_action(self, state, time):
        game = state.get_game()
        root = self.mcts_search(state)
        best = root.best_child()
        mcts_action = best.action
        return mcts_action

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



class SimpleEvaluator(Evaluator):
    def __init__(self):
        pass

    def evaluate(self, state):
        """Returns a simple evaluation of the given state."""
        return state.returns()

    def prior(self, state):
        """Returns a uniform prior probability over all legal actions."""
        actions = state.legal_actions()
        num_actions = len(actions)
        return [(action, 1/num_actions) for action in actions]  # Uniform probability


class RandomRolloutEvaluator(Evaluator):
  def __init__(self, n_rollouts=1, random_state=None):
    self.n_rollouts = n_rollouts
    self._random_state = random_state or np.random.RandomState()

  def evaluate(self, state):
    result = None
    for _ in range(self.n_rollouts):
      working_state = state.clone()
      while not working_state.is_terminal():
        if working_state.is_chance_node():
          outcomes = working_state.chance_outcomes()
          action_list, prob_list = zip(*outcomes)
          action = self._random_state.choice(action_list, p=prob_list)
        else:
          action = self._random_state.choice(working_state.legal_actions())
        working_state.apply_action(action)
      returns = np.array(working_state.returns())
      result = returns if result is None else result + returns

    return result / self.n_rollouts

  def prior(self, state):
    if state.is_chance_node():
      return state.chance_outcomes()
    else:
      legal_actions = state.legal_actions(state.current_player())
      return [(action, 1.0 / len(legal_actions)) for action in legal_actions]


def get_agent_for_tournament(player_id):
    game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})
    logging.info(f"MCTSAgent V6 created for tournament with player ID: {player_id}")
    num_simulations = 10
    exploration_coefficient = 1.414
    evaluator = RandomRolloutEvaluator(5)
    return LocalMCTSAgent(game, exploration_coefficient, num_simulations, evaluator)


