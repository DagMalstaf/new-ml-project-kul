import pyspiel
import random
import logging
import yaml
import os
import numpy as np

from open_spiel.python.algorithms.mcts import SearchNode
from task4.agent.mcts_openspiel.smart_agent_for_nn import *
from task4.training.training_alphazero.evaluator_V2 import AlphaZeroEvaluator
from task4.training.training_alphazero.resnet_model_V0 import NNetWrapper as nn


center_moves = [16, 74, 17, 75, 18, 76, 23, 77, 24, 82, 25, 83, 30, 84, 31, 85, 32, 90, 37, 91, 38, 92, 39, 93]
spokes_moves = [0, 57, 6, 62, 42, 105, 48, 110]

game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})
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

class MCTSAgent(pyspiel.Bot):
    def __init__(self, game, uct_c, max_simulations, evaluator, solve=True,
                 random_state=None, child_selection_fn=SearchNode.uct_value,
                 dirichlet_noise=None, verbose=False, dont_return_chance_node=False):
        pyspiel.Bot.__init__(self)

        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError("Game must have terminal rewards.")
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("Game must have sequential turns.")
        
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

        
    def restart_at(self, state):
        pass

    def step_with_policy(self, state, temp=1):
        root, count = self.mcts_search(state)
        self.simulations_count = count
        best = root.best_child()
    
        mcts_action = best.action

        if temp == 0:
            max_index = mcts_action
            return [1.0 if i == max_index else 0.0 for i in range(state.num_distinct_actions())], mcts_action

        total_counts = sum([child.explore_count for child in root.children])
        policy = [(child.action,  child.explore_count/total_counts) for child in root.children]

        for i in range(state.num_distinct_actions()):
            if i not in [x[0] for x in policy]:
                policy.append((i,0.0))

        policy = sorted(policy, key=lambda x: x[0])
        
        policy = [x[1] for x in policy]
        
        return policy, mcts_action

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

        chain_action, chain_action_available = protocol_moveonly(80, 80, all_moves, all_indices)
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

    def step(self, state, temp=0):
        state_custom_string = self.form_game_string(state)
        neg_moves = []
        pos_moves = []
        chain_action, chain_action_available = self.get_chain_action_if_present(state_custom_string)
        if (chain_action_available == True):
            policy = [(chain_action, 1.0)]
            for i in range(state.num_distinct_actions()):
                if i not in [x[0] for x in policy]:
                    policy.append((i,0.0))
            policy = sorted(policy, key=lambda x: x[0])
            policy = [x[1] for x in policy]
            return policy, chain_action
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
                policy = [(action, 1.0)]
                for i in range(state.num_distinct_actions()):
                    if i not in [x[0] for x in policy]:
                        policy.append((i,0.0))
                policy = sorted(policy, key=lambda x: x[0])
                policy = [x[1] for x in policy]
                return policy, action
            else:
                return self.step_with_policy(state, temp)

    def get_action(self,state, time):
        return self.step(state)

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

    def mcts_search(self, state):
        root = SearchNode(None, state.current_player(), 1)
        counter = 0
        for i in range(self.max_simulations):
            counter += 1
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

        return root, counter
  
def get_agent_for_tournament(player_id):
    config_path = 'task4/training/training_alphazero/config.yaml'
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
    
    nnet.load_checkpoint(config['checkpoint_folder'], config['checkpoint_file'])
    
    evaluator = AlphaZeroEvaluator(game, nnet)

    agent = MCTSAgent(game, config['exploration_coefficient'], config['num_MCTS_sims'], evaluator)

    logging.info(f"AlphaZero Agent created for tournament with player ID: {player_id}")
    return agent


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)