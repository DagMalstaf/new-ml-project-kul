import numpy as np
import pyspiel
import logging
import hashlib
import time
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing
import random

################################################################################################################################

class LocalMCTSAgent:
    def __init__(self, game, exploration_coefficient, initial_action=None):
        self.local_tree = {}
        self.local_simulations_count = 0
        self.game = game
        self.exploration_coefficient = exploration_coefficient
        self.initial_action = initial_action

    def hash_state(self, state):
        state_str = str(state)
        return hashlib.sha256(state_str.encode()).hexdigest()

    def run_simulation(self, state):
        self.local_simulations_count += 1
        path = []
        if self.initial_action is not None and self.initial_action in state.legal_actions():
            state_key = self.hash_state(state)
            self.local_tree[state_key] = self.populate_tree_with_new_state(state)
            state.apply_action(self.initial_action)
            path.append((state_key, self.initial_action))

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
        
        for state_key, action in reversed(path):
            action_idx = self.local_tree[state_key]['action_to_index'][action]
            self.local_tree[state_key]['N'][action_idx] += 1
            self.local_tree[state_key]['W'][action_idx] += state.returns()[0]
            if self.local_tree[state_key]['N'][action_idx] > 0:
                self.local_tree[state_key]['Q'][action_idx] = self.local_tree[state_key]['W'][action_idx] / self.local_tree[state_key]['N'][action_idx]
            else:
                raise Exception("Q is null")
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
        total_visits = np.sum(node['N']) + 1
        U = self.exploration_coefficient * np.sqrt(np.log(total_visits) / (1 + node['N']))
        values = node['Q'] + np.where(node['N'] > 0, U, float('inf'))
        best_action_index = np.argmax(values)
        return node['legal_actions'][best_action_index]
    
    
################################################################################################################################

class ExplorationEnhancedAgent(LocalMCTSAgent):
    def __init__(self, game, base_exploration_coefficient, initial_action=None):
        super().__init__(game, base_exploration_coefficient, initial_action)
        self.base_exploration_coefficient = base_exploration_coefficient
        self.initial_action = initial_action

    def select_action(self, state_key, tree):
        node = tree[state_key]
        total_visits = np.sum(node['N']) + 1

        # Dynamic exploration coefficient based on game phase
        exploration_coefficient = self.adjust_exploration_coefficient(total_visits)

        # Progressive widening: controls the number of actions considered based on total visits
        num_actions_to_consider = self.progressive_widening(node, total_visits)

        # Rare move boosting: encourages exploring less visited moves
        rare_move_bonus = self.rare_move_boosting(node)

        # Calculate the UCB values for action selection
        U = exploration_coefficient * np.sqrt(np.log(total_visits) / (1 + node['N'][:num_actions_to_consider])) + rare_move_bonus
        values = node['Q'][:num_actions_to_consider] + U
        best_action_index = np.argmax(values)
        return node['legal_actions'][best_action_index]

    def adjust_exploration_coefficient(self, total_visits):
        # Dynamically adjust exploration coefficient based on the stage of the game
        # For simplification, assuming the exploration coefficient decreases as the number of visits increases
        return self.base_exploration_coefficient / np.log(1 + total_visits)

    def progressive_widening(self, node, total_visits):
        k = 1.0  # Base constant for widening
        m = 0.5  # Power factor for widening (lower than 1 to be conservative)
        max_actions = len(node['legal_actions'])
        num_actions_to_consider = int(k * total_visits**m)
        return min(max_actions, num_actions_to_consider)

    def rare_move_boosting(self, node):
        # Adding a small bonus to less visited moves
        min_visits = np.min(node['N'])
        return 0.01 * (1 / (1 + node['N'] - min_visits))


################################################################################################################################

class EfficiencyEnhancedAgent(LocalMCTSAgent):
    def __init__(self, game, exploration_coefficient, initial_action=None, max_depth_factor=1.0):
        super().__init__(game, exploration_coefficient, initial_action)
        self.max_depth_factor = max_depth_factor 

    def run_simulation(self, state):
        self.local_simulations_count += 1
        path = []
        max_depth = self.calculate_max_depth(state)
        current_depth = 0
        
        if self.initial_action is not None and self.initial_action in state.legal_actions():
            state.apply_action(self.initial_action)
            state_key = self.hash_state(state)
            self.local_tree[state_key] = self.populate_tree_with_new_state(state)
            path.append((state_key, self.initial_action))
            current_depth += 1

        while not state.is_terminal(): # "and current_depth < max_depth"
            logging.info(state)
            state_key = self.hash_state(state)
            if state_key not in self.local_tree:
                self.local_tree[state_key] = self.populate_tree_with_new_state(state)

            action = self.select_action(state_key, self.local_tree)
            state.apply_action(action)
            path.append((state_key, action))
            current_depth += 1
            '''
            # Early pruning if the outcome becomes predictable
            if self.should_prune(state, path):
                break
            '''
            
        logging.info(state.returns()[0])
        for state_key, action in reversed(path):
            #logging.info("State key: %s, Action: %s" % (state_key,action))
            action_idx = self.local_tree[state_key]['action_to_index'][action]
            self.local_tree[state_key]['N'][action_idx] += 1
            self.local_tree[state_key]['W'][action_idx] += state.returns()[0]
            self.local_tree[state_key]['Q'][action_idx] = self.local_tree[state_key]['W'][action_idx] / self.local_tree[state_key]['N'][action_idx]
            
        logging.info(self.local_tree)
        logging.info(self.local_simulations_count)
        return self.local_tree, self.local_simulations_count

    def calculate_max_depth(self, state):
        """ Dynamically calculates the maximum depth of the simulation based on game state complexity """
        # Example: Less complex states could be defined by fewer remaining moves
        remaining_moves = len(state.legal_actions())
        return int(self.max_depth_factor * np.sqrt(remaining_moves))

    def should_prune(self, state, path):
        """ Determine if the simulation should be pruned early based on outcome predictability """
        if len(path) > 5:  # Arbitrary minimum depth before pruning can occur
            recent_rewards = [state.returns()[0] for _, _ in path[-5:]]
            if np.std(recent_rewards) < 0.1:  # Low variance in the recent rewards suggests predictability
                return True
        return False


################################################################################################################################

class DecisionEnhancedAgent(LocalMCTSAgent):
    def __init__(self, game, exploration_coefficient, initial_action=None):
        super().__init__(game, exploration_coefficient, initial_action)
        self.history = {}  # To track historical performance of actions

    def run_simulation(self, state):
        self.local_simulations_count += 1
        path = []
        
        if self.initial_action is not None and self.initial_action in state.legal_actions():
            state.apply_action(self.initial_action)
            state_key = self.hash_state(state)
            self.local_tree[state_key] = self.populate_tree_with_new_state(state)
            path.append((state_key, self.initial_action))

        while not state.is_terminal():
            state_key = self.hash_state(state)
            if state_key not in self.local_tree:
                self.local_tree[state_key] = self.populate_tree_with_new_state(state)

            action = self.select_action(state_key, self.local_tree, state)
            state.apply_action(action)
            path.append((state_key, action))
        
        reward = state.returns()[0]
        self.backpropagate(path, reward)

        return self.local_tree, self.local_simulations_count

    def select_action(self, state_key, tree, current_state):
        node = tree[state_key]
        total_visits = np.sum(node['N']) + 1
        
        # Action prioritization based on historical performance
        historical_scores = self.calculate_historical_scores(node['legal_actions'])
        
        U = self.exploration_coefficient * np.sqrt(np.log(total_visits) / (1 + node['N']))
        values = node['Q'] + U + historical_scores
        best_action_index = np.argmax(values)
        return node['legal_actions'][best_action_index]

    def calculate_historical_scores(self, actions):
        # Fetch historical performance and adjust scores
        scores = np.array([self.history.get(action, 0) for action in actions])
        return scores / (np.max(scores) + 1)  # Normalize to prevent overpowering exploration

    def backpropagate(self, path, final_reward):
        # Enhanced backpropagation with reward scaling and TD-learning
        for state_key, action in reversed(path):
            action_idx = self.local_tree[state_key]['action_to_index'][action]
            if action_idx in self.local_tree[state_key]['N']:
                self.local_tree[state_key]['N'][action_idx] += 1
                # Temporal difference learning component
                immediate_reward = self.scale_reward(state_key, final_reward)
                self.local_tree[state_key]['W'][action_idx] += immediate_reward
                self.local_tree[state_key]['Q'][action_idx] = self.local_tree[state_key]['W'][action_idx] / self.local_tree[state_key]['N'][action_idx]

                # Update historical performance
                self.history[action] = self.history.get(action, 0) + immediate_reward

    def scale_reward(self, state_key, reward):
        # Scale reward based on the game state's significance
        # This could be more sophisticated based on game-specific metrics
        return reward * (1 + np.log(1 + self.local_tree[state_key]['N'].sum()))


################################################################################################################################

class HeuristicEnhancedAgent(LocalMCTSAgent):
    def __init__(self, game, exploration_coefficient, initial_action=None):
        super().__init__(game, exploration_coefficient, initial_action)

    def select_action(self, state_key, tree):
        node = tree[state_key]
        total_visits = np.sum(node['N']) + 1

        # Incorporate heuristic evaluations into the decision process
        heuristic_scores = self.evaluate_heuristics(state_key, node['legal_actions'])

        # Calculate the Upper Confidence Bound for Trees (UCT)
        U = self.exploration_coefficient * np.sqrt(np.log(total_visits) / (1 + node['N']))
        values = node['Q'] + U + heuristic_scores
        best_action_index = np.argmax(values)
        return node['legal_actions'][best_action_index]

    def evaluate_heuristics(self, state_key, actions):
        """ Evaluate game-specific heuristics to adjust action values based on the game state's strategic elements. """
        scores = np.zeros(len(actions))
        current_state = self.game.get_state_from_key(state_key)  # Hypothetical method to reconstruct state from key

        for i, action in enumerate(actions):
            hypothetical_state = current_state.clone()
            hypothetical_state.apply_action(action)
            scores[i] = self.chain_heuristic(hypothetical_state)

        return scores

    def chain_heuristic(self, state):
        """ Heuristic that evaluates the value of forming or extending chains. """
        # Score states higher if they result in capturing a chain or setting up for future captures
        # Placeholder logic: count the potential number of boxes completed by a move
        score = 0
        for box in state.get_boxes():  # Hypothetical method to check box completion status
            if box.is_completed():
                score += 1
            elif box.is_almost_completed():  # One line missing to complete
                score -= 0.5  # Penalize potentially leaving an easy capture for the opponent
        return score

    def backpropagate(self, path, final_reward):
        """ Override to incorporate heuristic adjustments in the backpropagation phase. """
        for state_key, action in reversed(path):
            action_idx = self.local_tree[state_key]['action_to_index'][action]
            self.local_tree[state_key]['N'][action_idx] += 1
            adjusted_reward = self.adjust_reward_based_on_heuristics(state_key, final_reward)
            self.local_tree[state_key]['W'][action_idx] += adjusted_reward
            self.local_tree[state_key]['Q'][action_idx] = self.local_tree[state_key]['W'][action_idx] / self.local_tree[state_key]['N'][action_idx]

    def adjust_reward_based_on_heuristics(self, state_key, reward):
        """ Adjust rewards based on the strategic importance of the resulting state. """
        # This could adjust the reward based on the number of chains closed, the game phase, etc.
        return reward


################################################################################################################################

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
            initial_actions = [random.choice(state.legal_actions()) for _ in range(self.process_pool.ncpus)]
            logging.info("Initial actions: %s", initial_actions)
            agents = [EfficiencyEnhancedAgent(self.game, self.exploration_coefficient, initial_action=ia) for ia in initial_actions]
            '''
            agents = [
                ExplorationEnhancedAgent(self.game, self.exploration_coefficient, initial_action=initial_actions[0]),
                EfficiencyEnhancedAgent(self.game, self.exploration_coefficient, initial_action=initial_actions[1]),
                DecisionEnhancedAgent(self.game, self.exploration_coefficient, initial_action=initial_actions[2]),
                HeuristicEnhancedAgent(self.game, self.exploration_coefficient, initial_action=initial_actions[3])
            ]
            '''
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
    logging.info(f"MCTSAgent V4 created for tournament with player ID: {player_id}")
    num_simulations = 10
    exploration_coefficient = 1.41
    return MCTSAgent(game, num_simulations, exploration_coefficient)


