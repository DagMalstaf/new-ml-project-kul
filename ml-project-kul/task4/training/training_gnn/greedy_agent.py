                                                                                                         
#/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxes_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022-2024 KU Leuven. All rights reserved.
"""

import sys
import argparse
import logging
import random
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots


logger = logging.getLogger('be.kuleuven.cs.dtai.dotsandboxes')


def get_agent_for_tournament_greedy(player_id, evaluator):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id, evaluator)
    return my_player


def part2num(part):
    p = {'h': 0, 'horizontal': 0,  # Who has set the horizontal line (top of cell)
         'v': 1, 'vertical':   1,  # Who has set the vertical line (left of cell)
         'c': 2, 'cell':       2}  # Who has won the cell
    return p.get(part, part)


def state2num(state):
    s = {'e':  0, 'empty':   0,
         'p1': 1, 'player1': 1,
         'p2': 2, 'player2': 2}
    return s.get(state, state)


def num2state(state):
    s = {0: 'empty', 1: 'player1', 2: 'player2'}
    return s.get(state, state)


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id, evaluator):
        """Initialize an agent to play Dots and Boxes.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)
        self.player_id = player_id
        self.num_cols = 7
        self.num_rows = 7
        self.num_cells = (self.num_rows + 1) * (self.num_cols + 1)
        self.num_parts = 3
        self.game = None
        self.evaluator = evaluator

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        self.game = state.get_game()
        params = self.game.get_parameters()
        self.num_cols = params['num_cols']
        self.num_rows = params['num_rows']
        self.num_cells = (self.num_rows + 1) * (self.num_cols + 1)
        print(f"Restart game with params {params}")

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        pass

    
    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        if self.game is None:
            self.game = state.get_game()

        legal_actions = state.legal_actions()
        policy = self.evaluator.policy(state)


        for r in range(self.num_rows):
            for c in range(self.num_cols):
                cnt = 0
                a = None
                for rd, cd, o in [(0,0,'h'), (0,0,'v'), (0,1,'v'), (1,0,'h')]:
                    s = self.get_observation_state(state.observation_tensor(), r+rd, c+cd, o)
                    if s != 'empty':
                        cnt += 1
                    else:
                        a = (r+rd, c+cd, o)
                if cnt == 3:
                    if a[2] == 'h':
                        action = a[0]*self.num_cols + a[1]
                    else:
                        action = self.num_cols*(self.num_rows+1) + a[0]*(self.num_cols+1) + a[1]
                    if action not in legal_actions:
                        print(f"ERROR: illegal action chosen: {action} is not in {legal_actions}")
                    else:
                        return policy, action
        

        rand_idx = random.randint(0, len(legal_actions) - 1)
        action = legal_actions[rand_idx]
        return policy, action

    def get_policy_vector(self, state):
        num_actions = self.game.num_distinct_actions()
        policy_vector = np.zeros(num_actions)

        state_input = self.convert_state_to_input(state)

        action_probabilities = self.model.predict(state_input)

        # Only assign probabilities to legal actions
        legal_actions = state.legal_actions()
        for action in legal_actions:
            policy_vector[action] = action_probabilities[action]

        # Normalize the policy vector
        policy_vector /= np.sum(policy_vector)
        
        return policy_vector

    def convert_state_to_input(self, state):
        """Convert the game state to a format suitable for neural network input."""
        # Implement the conversion logic based on your model's input requirements
        pass


    def get_observation(self, obs_tensor, state, row, col, part):
        state = state2num(state)
        part = part2num(part)
        idx =   part \
              + (row * (self.num_cols + 1) + col) * self.num_parts  \
              + state * (self.num_parts * self.num_cells)
        return obs_tensor[idx]


    def get_observation_state(self, obs_tensor, row, col, part, as_str=True):
        is_state = None
        for state in range(3):
            if self.get_observation(obs_tensor, state, row, col, part) == 1.0:
                is_state = state
        if as_str:
            is_state = num2state(is_state)
        return is_state

def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    dotsandboxes_game_string = (
        "dotsandboxes(num_rows=5,num_cols=5)")
    game = pyspiel.load_game(dotsandboxes_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0,1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())