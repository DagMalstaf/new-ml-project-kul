#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import pyspiel
import random
import logging
from smart_agent_for_nn import *
from open_spiel.python.algorithms.mcts import MCTSBot, SearchNode, RandomRolloutEvaluator
log = logging.getLogger(__name__)

class MCTSAgent():
    def __init__(self, uct_c, max_simulations, evaluator, solve=True,
                 random_state=None, child_selection_fn=SearchNode.uct_value,
                 dirichlet_noise=None, verbose=False, dont_return_chance_node=False):
        self.game = None
        self.num_rows = None
        self.num_cols = None
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
        self.max_utility = None
        self.times = []
        self.steps = 0
        self.center_moves = None
        self.spokes_moves = None
        self.distinct_moves = None
        self.player_0_index_to_action = {}
        self.player_1_index_to_action = None
        self.bot = None


    def get_average_time(self):
        self.average_time = sum(self.times)/self.steps
        return self.average_time
    
    def step(self, state, temp=0):
        if self.game is None:
            self.game = state.get_game()
            self.num_rows = self.game.get_parameters()["num_rows"]
            self.num_cols = self.game.get_parameters()["num_cols"]
            self.distinct_moves = self.game.num_distinct_actions()
            self.max_utility = self.game.max_utility()
            self.bot = MCTSBot(self.game, self.uct_c, self.max_simulations, self.evaluator)

        if self.num_rows == self.num_cols and self.num_rows <= 7:
            self.setup()
            if self.center_moves is None:
                self.center_moves, self.spokes_moves = self.get_center_and_spokes_moves(self.num_cols)

            state_custom_string = self.form_game_string(state)
            neg_moves = []
            chain_action, chain_action_available = self.get_chain_action_if_present(state_custom_string, state)
            chain_action_available = False

            if chain_action_available:
                return chain_action
            else:
                prob = random.random()
                if prob > 0.9:
                    action, action_available = self.get_spokes_move(state_custom_string, neg_moves)
                    if not action_available:
                        action, action_available = self.get_center_move(state_custom_string, neg_moves)
                else:
                    action, action_available = self.get_center_move(state_custom_string, neg_moves)
                    if not action_available:
                        action, action_available = self.get_spokes_move(state_custom_string, neg_moves)

                if action_available:
                    return action
                else:
                    action = self.step_with_policy(state, temp)
                    return action
        else:
            action = self.step_with_policy(state, temp)
            return action

    def get_chain_action_if_present(self, state_custom_string, state):
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
        for i in range(0, len(self.spokes_moves)):
            if (state_custom_string[self.spokes_moves[i]] == '0' and self.spokes_moves[i] not in neg_moves):
                return self.spokes_moves[i], True
        return self.spokes_moves[i], False

    def get_center_move(self, state_custom_string, neg_moves):
        for i in range(0, len(self.center_moves)):
            if (state_custom_string[self.center_moves[i]] == '0' and self.center_moves[i] not in neg_moves):
                return self.center_moves[i], True
        return self.center_moves[i], False
    
    def get_filled_edges(self, state_custom_string):
        filled_edges = self.distinct_moves
        for i in range(0, len(state_custom_string)):
            if (state_custom_string[i] == '0'):
                filled_edges = filled_edges - 1
        return filled_edges  
    
    def step_with_policy(self, state, temp=1):
        return self.bot.step(state)

    def form_game_string(self, state):
        state_custom_string = "0" * self.game.num_distinct_actions()
        state_custom_list = list(state_custom_string)
        for key in self.player_0_index_to_action.keys():
            tensor = state.observation_tensor()
            if tensor[key] == 1:
                state_custom_list[self.player_0_index_to_action[key]] = "1"

        for key in self.player_1_index_to_action.keys():
            tensor = state.observation_tensor()
            if tensor[key] == 1:
                state_custom_list[self.player_1_index_to_action[key]] = "2"

        state_custom_string = "".join(state_custom_list)
        return state_custom_string

    def setup(self):
        for i in range(0, self.distinct_moves):
            state = self.game.new_initial_state()
            initial_tensor = state.observation_tensor()
            state.apply_action(i)
            final_tensor = state.observation_tensor()
            for j in range(self.game.observation_tensor_size()//3, len(initial_tensor)):
                if (initial_tensor[j] != final_tensor[j]):
                    self.player_0_index_to_action[j] = i
        value = int(2/3 * (self.game.observation_tensor_size()))
        self.player_1_index_to_action = {value: 0}

        for i in range(1, self.distinct_moves):
            state = self.game.new_initial_state()
            state.apply_action(0)
            initial_tensor = state.observation_tensor()
            state.apply_action(i)
            final_tensor = state.observation_tensor()
            for j in range(self.game.observation_tensor_size()//3, len(initial_tensor)):
                if (initial_tensor[j] != final_tensor[j]):
                    self.player_1_index_to_action[j] = i

    def get_center_and_spokes_moves(self, n):
        center_moves = []
        tot = 2 * n * (n + 1)
        i_divs = []
        j_divs = []
        if (n%2 == 0):
            lines = (n - 1)//2
            for i in range(lines):
                i_divs.append([n//2])

            for i in range(lines):
                j_divs.append([(n - 1)//2 - i, n - ((n -- 1)//2 - i)])
        else:
            boxes = n//2
            for times in range(0, boxes):
                i_divs.append([boxes - times, n - boxes + times])

            for times in range(0, boxes):  
                current_div = []
                for index in range(boxes - times, boxes + times + 1):
                    current_div.append(index)
                j_divs.append(current_div)

        for z in range(0, len(i_divs)):
            for i in i_divs[z]:
                for j in j_divs[z]:
                    center_moves.append(n * i + j)
            for i in i_divs[z]:
                for j in j_divs[z]:
                    center_moves.append(tot//2 + (n + 1) * j + i)
        spokes_moves = [n, tot//2 + 1, 2*n - 1, tot//2 + n - 1, tot//2 - 2*n,  tot - n, tot//2 - n - 1, tot - n + (n - 2)]
        return center_moves, spokes_moves
    
    def get_long_chain_action(self, state_custom_string, state):
        components = get_components(strat_graph)
        long_chains = [c for c in components if len(c) >= 3]

        for chain in long_chains:
            if chain_is_open(chain, self.strat_box_dict):
                return take_chain(chain, self.strat_box_dict)

        return None, False

    def get_open_chain_action(self, state_custom_string, state):
        components = get_components(strat_graph)
        short_chains = [c for c in components if len(c) == 2]

        for chain in short_chains:
            if not chain_is_open(chain, self.strat_box_dict):
                return open_chain(chain, self.strat_box_dict)

        return None, False

    def get_action(self,state, time):
        return self.step(state)

    def restart_at(self, state):
        pass

    def inform_action(self, state, player_id, action):
        pass

def get_agent_for_tournament(player_id):
    num_simulations = 20
    exploration_coefficient = 1.414
    evaluator = RandomRolloutEvaluator(3)
    agent = MCTSAgent(exploration_coefficient, num_simulations, evaluator)
    print(f"MCTS Agent created for tournament with player ID: {player_id}")
    return agent






