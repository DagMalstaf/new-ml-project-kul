#!/usr/bin/env python3
# encoding: utf-8

import random
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator, MCTSBot


class Agent(pyspiel.Bot):
    def __init__(self, player_id):
        pyspiel.Bot.__init__(self)
        self.player_id = player_id

    def restart_at(self, state):
        pass

    def inform_action(self, state, player_id, action):
        pass

    def step(self, state):
        legal_actions = state.legal_actions()
        rand_idx = random.randint(0, len(legal_actions) - 1)
        action = legal_actions[rand_idx]
        return action
    

class MCTSAgent(Agent):
    def __init__(self, player_id):
        super().__init__(player_id)
        dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=7,num_cols=7)")
        self.game = pyspiel.load_game(dotsandboxes_game_string)
        self.bot = MCTSBot(self.game, 1.41, 20, RandomRolloutEvaluator(5))
    
    def restart_at(self, state):
        pass

    def inform_action(self, state, player_id, action):
        pass

    def step(self, state):
        return self.bot.step(state)

def get_agent_for_tournament(player_id):

    print('Created random agent for tournament')
    return MCTSAgent(player_id)
