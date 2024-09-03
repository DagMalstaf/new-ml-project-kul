import pyspiel
import random
import logging
import yaml
import os
import numpy as np
import time
import signal
import functools
import errno

from open_spiel.python.algorithms.mcts import SearchNode
from SmartMovesAgent import *
from NeuralEvaluator import AlphaZeroEvaluator
from AlphaZeroModel import NNetWrapper as nn
game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})


def get_center_spokes_moves(n):
    if (n == 2):
        return [], []
    elif (n == 3):
        return [4, 17, 7, 18], [3, 13, 5, 14, 6, 21, 8, 22]
    elif (n == 4):
        return [9, 27, 10, 32], [4, 21, 7, 23, 12, 36, 15, 38]
    elif (n == 5):
        return [12, 44, 17, 45, 6, 37, 7, 40, 8, 43, 21, 46, 22, 49, 23, 52], [5, 31, 9, 34, 20, 55, 24, 58]
    elif (n == 6):
        return [20, 59, 21, 66, 19, 52, 22, 73], [6, 43, 11, 47, 30, 78, 35, 82]
    elif (n == 7):
        return [24, 83, 31, 84, 16, 74, 17, 77, 18, 82, 37, 85, 38, 90, 39, 93], [7, 57, 13, 62, 42, 105, 48, 110]

center_moves, spokes_moves = get_center_spokes_moves(7)
        
player_0_index_to_action = {}
for i in range(0, 112):
    state = game.new_initial_state()
    initial_tensor = state.observation_tensor()
    state.apply_action(i)
    final_tensor = state.observation_tensor()
    for j in range(576//3, len(initial_tensor)):
        if (initial_tensor[j] != final_tensor[j]):
            player_0_index_to_action[j] = i
    
player_1_index_to_action = {384 : 0}
for i in range(1, 112):
    state = game.new_initial_state()
    state.apply_action(0)
    initial_tensor = state.observation_tensor()
    state.apply_action(i)
    final_tensor = state.observation_tensor()
    for j in range(576//3, len(initial_tensor)):
        if (initial_tensor[j] != final_tensor[j]):
            player_1_index_to_action[j] = i


class AlphaZeroAgent():
    def __init__(self, game, uct_c, max_simulations, evaluator, solve=True,
                 random_state=None, child_selection_fn=SearchNode.uct_value,
                 dirichlet_noise=None, verbose=False, dont_return_chance_node=False):
        self._game = game
        self.uct_c = uct_c
        self.max_simulations = max_simulations
        self.evaluator = evaluator
        self.verbose = verbose
        self.solve = solve
        self._dirichlet_noise = dirichlet_noise
        self._random_state = random_state or np.random.RandomState()
        self._child_selection_fn = child_selection_fn
        self.dont_return_chance_node = dont_return_chance_node
        self.num_processes = 4
        self.process_pool = None
        self.simulations_count = 0
        self.max_utility = self._game.max_utility()
        self.times = []
        self.steps = 0
    
    def get_average_time(self):
        self.average_time = sum(self.times)/self.steps
        return self.average_time
    
    def step(self, state, temp=0):
        start_time = time.time()
        state_custom_string = self.form_game_string(state)
        neg_moves = []
        chain_action, chain_action_available = self.get_chain_action_if_present(state_custom_string)
        chain_action_available = False

        if (chain_action_available == True):
            return chain_action
        else:
            prob = random.random()
            if (prob > 0.9):
                action, action_available = self.get_spokes_move(state_custom_string, neg_moves)
                if (action_available == False):
                    action, action_available = self.get_center_move(state_custom_string, neg_moves)
            else:
                action, action_available = self.get_center_move(state_custom_string, neg_moves)
                if (action_available == False):
                    action, action_available = self.get_spokes_move(state_custom_string, neg_moves)
            if (action_available == True):
                return action
            else:
                action = self.step_with_policy(state, temp)
                end_time = time.time()
                self.steps += 1
                self.times.append(end_time - start_time)     
                return action

    def get_chain_action_if_present(self, state_custom_string):
        all_moves = []
        all_indices = []
        for i in range(0, len(state_custom_string)):
            if (state_custom_string[i] == '1'):
                all_indices.append(1)
                action = i
                if (action < 56):
                    all_moves.append(((action//7, action%7), (action//7, action%7 + 1)))
                else:
                    all_moves.append((((action - 56)%7, (action - 56)//7), ((action - 56)%7 + 1, (action - 56)//7)))
            if (state_custom_string[i] == '2'):
                all_indices.append(2)
                action = i
                if (action < 56):
                    all_moves.append(((action//7, action%7), (action//7, action%7 + 1)))
                else:
                    all_moves.append((((action - 56)%7, (action - 56)//7), ((action - 56)%7 + 1, (action - 56)//7)))

        chain_action, chain_action_available = protocol_moveonly(80, 80, all_moves, all_indices, int(state.current_player()) + 1)
        chain_action = edge_to_spiel_action(chain_action)
        return chain_action, chain_action_available

    def get_spokes_move(self, state_custom_string, neg_moves):
        for i in range(0, len(spokes_moves)):
            if (state_custom_string[spokes_moves[i]] == '0' and spokes_moves[i] not in neg_moves):
                return spokes_moves[i], True
        return spokes_moves[i], False

    def get_center_move(self, state_custom_string, neg_moves):
        for i in range(0, len(center_moves)):
            if (state_custom_string[center_moves[i]] == '0' and center_moves[i] not in neg_moves):
                return center_moves[i], True
        return center_moves[i], False
    
    def get_filled_edges(self, state_custom_string):
        filled_edges = 112
        for i in range(0, len(state_custom_string)):
            if (state_custom_string[i] == '0'):
                filled_edges = filled_edges - 1
        return filled_edges  
    
    def step_with_policy(self, state, temp=1):
        root = self.mcts_search(state)
        best = root.best_child()
        return best.action
    
    def mcts_search(self, state):
        root = SearchNode(None, state.current_player(), 1)
        for _ in range(self.max_simulations):
            self.simulations_count += 1
            visit_path, working_state = self._apply_tree_policy(root, state)
            if working_state.is_terminal():
                returns = working_state.returns()
                visit_path[-1].outcome = returns
                solved = self.solve
            else:
                returns = self.evaluator.evaluate(working_state)
                solved = False

            while visit_path:
                decision_node_idx = -1
                while visit_path[decision_node_idx].player == pyspiel.PlayerId.CHANCE:
                    decision_node_idx -= 1
                target_return = returns[visit_path[decision_node_idx].player]
                node = visit_path.pop()
                node.total_reward += target_return
                node.explore_count += 1

                if solved and node.children:
                    player = node.children[0].player
                    if player == pyspiel.PlayerId.CHANCE:
                        outcome = node.children[0].outcome
                        if (outcome is not None and all(np.array_equal(c.outcome, outcome) for c in node.children)):
                            node.outcome = outcome
                        else:
                            solved = False
                    else:
                        best = None
                        all_solved = True
                        for child in node.children:
                            if child.outcome is None:
                                all_solved = False
                            elif best is None or child.outcome[player] > best.outcome[player]:
                                best = child
                        if (best is not None and (all_solved or best.outcome[player] == self.max_utility)):
                            node.outcome = best.outcome
                        else:
                            solved = False
            if root.outcome is not None:
                break

        return root
  
    def _apply_tree_policy(self, root, state):
        visit_path = [root]
        working_state = state.clone()
        current_node = root
        while (not working_state.is_terminal() and current_node.explore_count > 0) or ( working_state.is_chance_node() and self.dont_return_chance_node):
            if not current_node.children:
                legal_actions = self.evaluator.prior(working_state)
                if current_node is root and self._dirichlet_noise:
                    epsilon, alpha = self._dirichlet_noise
                    noise = self._random_state.dirichlet([alpha] * len(legal_actions))
                    legal_actions = [(a, (1 - epsilon) * p + epsilon * n) for (a, p), n in zip(legal_actions, noise)]
                
                self._random_state.shuffle(legal_actions)
                player = working_state.current_player()
                current_node.children = [ SearchNode(action, player, prior) for action, prior in legal_actions ]

            if working_state.is_chance_node():
                outcomes = working_state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = self._random_state.choice(action_list, p=prob_list)
                chosen_child = next(c for c in current_node.children if c.action == action)
            else:
                chosen_child = max( current_node.children, key=lambda c: self._child_selection_fn(c, current_node.explore_count, self.uct_c))

            working_state.apply_action(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)

        return visit_path, working_state

    
    def form_game_string(self, state):
        state_custom_string = "0" * 112
        state_custom_list = list(state_custom_string)
        for key in player_0_index_to_action.keys():
            tensor = state.observation_tensor()
            if tensor[key] == 1:
                state_custom_list[player_0_index_to_action[key]] = "1"

        for key in player_1_index_to_action.keys():
            tensor = state.observation_tensor()
            if tensor[key] == 1:
                state_custom_list[player_1_index_to_action[key]] = "2"

        state_custom_string = "".join(state_custom_list)
        return state_custom_string

    def get_action(self,state, time):
        return self.step(state)

    def restart_at(self, state):
        pass

    def inform_action(self, state, player_id, action):
        pass
  
def get_agent_for_tournament(player_id):
    current_directory = os.path.dirname(__file__)
    config_path = os.path.join(current_directory, 'config.yaml')
    config = load_config(config_path)
    
    num_rows = config['num_rows']
    num_cols = config['num_cols']
    game_string = f'dots_and_boxes(num_rows={num_rows},num_cols={num_cols})'
    game = pyspiel.load_game(game_string)

    nnet = nn(game,
              config['checkpoint_path'],
              config['learning_rate'],
              config['nn_width'],
              config['nn_depth'],
              config['weight_decay'],
              config)
    
    nnet.load_checkpoint(current_directory, config['tournament_file_name'])
    
    evaluator = AlphaZeroEvaluator(game, nnet)

    agent = AlphaZeroAgent(game, config['exploration_coefficient'], config['numMCTSSimsTournament'], evaluator, player_id)

    print(f"AlphaZero Agent created for tournament with player ID: {player_id}")
    return agent


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)