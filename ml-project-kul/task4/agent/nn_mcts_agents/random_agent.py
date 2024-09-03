#!/usr/bin/env python3
# encoding: utf-8

import random
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots
import logging


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
    

class RandomAgent(Agent):
    def __init__(self, player_id):
        super().__init__(player_id)
        self.simulations_count = 0
    
    def restart_at(self, state):
        pass

    def inform_action(self, state, player_id, action):
        pass
    
    def get_action(self, state, time_limit):
        return self.step(state)

    def step(self, state):
        legal_actions = state.legal_actions()
        if not legal_actions:
            return pyspiel.INVALID_ACTION 
        self.simulations_count += 1
        return random.choice(legal_actions)

def get_agent_for_tournament(player_id):
    logging.info(f"Random Agent created for tournament with player ID: {player_id}")
    return RandomAgent(player_id)

