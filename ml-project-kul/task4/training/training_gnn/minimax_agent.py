#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxes_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""

import sys
import logging
import time
import os
import numpy as np
import pyspiel

package_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(package_directory, '../')
sys.path.append(path)

from open_spiel.python.algorithms import evaluate_bots


from task4.training.training_gnn.source.transposition_table import TOptimised_Table, Transposition_Table_Chains
from task4.training.training_gnn.source.chains_strategy import StrategyAdvisor
from task4.training.training_gnn.source.evaluators import eval_maximize_difference
from task4.training.training_gnn.source.alphabeta import minimax_alphabeta_search



logger = logging.getLogger('be.kuleuven.cs.dtai.dotsandboxes')

def get_agent_for_tournament_minimax(player_id, evaluator):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id, evaluator)
    return my_player


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
        self.TTC = Transposition_Table_Chains()
        self.SA = None
        self.evaluator = evaluator

    def restart_at(self, state):
        """Starting a new game in the given state.
        :param state: The initial state of the game.
        """
        params = state.get_game().get_parameters()
        num_rows = params['num_rows']
        num_cols = params['num_cols']
        self.SA = StrategyAdvisor(num_rows, num_cols)
        self.TT = TOptimised_Table(num_rows, num_cols)

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        if self.SA is None: 
            logger.info("self.SA is None in inform_action")
            self.restart_at(state)

        if player_id != self.player_id:
            self.SA.update_action(action, state.current_player())

    def step(self, state):
        if self.SA is None: 
            logger.info("self.SA is None in step")
            self.restart_at(state)

        max_allowed_depth = 9
        
        _, best_action = minimax_alphabeta_search(game=state.get_game(),
                                            state=state.clone(),
                                            transposition_table=self.TT,
                                            transposition_table_chains=self.TTC,
                                            strategy_advisor=self.SA,
                                            maximum_depth=max_allowed_depth,
                                            value_function=eval_maximize_difference,

                                            maximizing_player_id=self.player_id)

        self.SA.update_action(best_action, state.current_player())

        policy = self.evaluator.prior(state)

        return policy, best_action



def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=5,num_cols=5)")
    
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

