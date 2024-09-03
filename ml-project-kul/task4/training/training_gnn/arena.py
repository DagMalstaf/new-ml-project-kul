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
            result = bot.step(state)
            if isinstance(result, tuple) and len(result) == 2:
                action = result[1]
            else:
                action = result 
            state.apply_action(action)
        
        return state.rewards() 
    
    def playGameWithExamples(self):
        trainExamples = []

        state = self.game.new_initial_state()
        curPlayer = state.current_player()
        episodeStep = 0
        
        players = [self.player1, self.player2]

        while not state.is_terminal():
            episodeStep += 1
            temp = int(episodeStep <= 35)
            
            curPlayer = state.current_player()

            bot = players[curPlayer]
            policy_vector, action = bot.step_with_policy(state)
            if isinstance(policy_vector[0], tuple) and len(policy_vector[0]) == 2:
                policy_vector = [prob for _, prob in policy_vector]

            trainExamples.append([np.array(state.observation_tensor()), policy_vector, state.current_player(), None])
            state.apply_action(action)

        reward = state.rewards()[curPlayer]
        trainExamples = [(x[0], x[1], reward * ((-1) ** (x[2] != curPlayer))) for x in trainExamples]
        return state.rewards(), trainExamples

    def playGames(self, num, verbose=False):
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult[0] == 1.0:
                oneWon += 1
            elif gameResult[1] == 1.0:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult[1] == 1.0:
                oneWon += 1
            elif gameResult[0] == 1.0:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
    
    def playGamesAgainstRandom(self, player1, num, verbose=False):
        rp2 = UniformRandomBot(1,np.random)
        arena = Arena(player1, rp2, self.game)
        oneWon, twoWon, draws = 0,0,0
        trainExamplesRandom = []
        
        for i in tqdm(range(num//2), desc="Playing_games_1"):
            reward, gameExamples = arena.playGameWithExamples()
            trainExamplesRandom.extend(gameExamples)
            if reward[0] == 1.0:
                oneWon += 1
            elif reward[1] == 1.0:
                twoWon += 1
            else:
                draws += 1
        
        rp2 = UniformRandomBot(0,np.random)
        arena = Arena(rp2,player1, self.game)

        for _ in tqdm(range(num//2), desc="Playing_games_2"):
            reward, gameExamples = arena.playGameWithExamples()
            trainExamplesRandom.extend(gameExamples)
            if reward[0] == 1.0:
                twoWon += 1
            elif reward[1] == 1.0:
                oneWon += 1
            else:
                draws += 1
        return oneWon, twoWon, draws, trainExamplesRandom
    

    def playGamesAgainstMCTS(self, player1, num, verbose=False):
        evaluator = RandomRolloutEvaluator(1, np.random)
        p2 = MCTSBot(self.game, 1, 25, evaluator)
        arena = Arena(player1, p2, self.game)
        oneWon, twoWon, draws = 0, 0, 0
        trainExamplesMCTS = []

        for _ in tqdm(range(num // 2), desc="Playing_games_1"):
            reward, gameExamples = arena.playGameWithExamples()
            trainExamplesMCTS.extend(gameExamples)
            if reward[0] == 1.0:
                oneWon += 1
            elif reward[1] == 1.0:
                twoWon += 1
            else:
                draws += 1

        arena = Arena(p2, player1, self.game)

        for _ in tqdm(range(num // 2), desc="Playing_games_2"):
            reward, gameExamples = arena.playGameWithExamples()
            trainExamplesMCTS.extend(gameExamples)
            if reward[0] == 1.0:
                twoWon += 1
            elif reward[1] == 1.0:
                oneWon += 1
            else:
                draws += 1
        
        return oneWon, twoWon, draws, trainExamplesMCTS
    
    
    
