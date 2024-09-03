import logging
log = logging.getLogger(__name__)
from tqdm import tqdm
from open_spiel.python.bots.uniform_random import UniformRandomBot
import numpy as np
from open_spiel.python.algorithms.mcts import MCTSBot
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator


class Arena():
    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        players = [self.player1, self.player2]
        state = self.game.new_initial_state()

        curPlayer = state.current_player()

        while not state.is_terminal():
            curPlayer = state.current_player()
          
            bot = players[curPlayer]
            result = bot.step_with_policy(state)
            if isinstance(result, tuple) and len(result) == 2:
                action = result[1]
            else:
                action = result 
            state.apply_action(action)
        
        return state.rewards() 
    
    def playGamesAgainstMCTS(self, player1, num, verbose=False):
        evaluator = RandomRolloutEvaluator(1, np.random)
        p2 = MCTSBot(self.game, 1, 25, evaluator)
        arena = Arena(player1, p2, self.game)
        oneWon, twoWon, draws = 0, 0, 0

        for _ in tqdm(range(num // 2), desc="Playing_games_1"):
            reward= arena.playGame()
            if reward[0] == 1.0:
                oneWon += 1
            elif reward[1] == 1.0:
                twoWon += 1
            else:
                draws += 1

        arena = Arena(p2, player1, self.game)

        for _ in tqdm(range(num // 2), desc="Playing_games_2"):
            reward = arena.playGame()
            if reward[0] == 1.0:
                twoWon += 1
            elif reward[1] == 1.0:
                oneWon += 1
            else:
                draws += 1
        
        return oneWon, twoWon, draws
    
    
    
