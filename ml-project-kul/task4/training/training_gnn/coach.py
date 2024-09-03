import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import csv
import torch

import numpy as np
from tqdm import tqdm
from task4.training.training_alphazero.resnet_model_V0 import get_distribution_strategy
from task4.training.training_gnn.arena import Arena
from task4.training.training_gnn.gnn_agent import MCTSAgent
from task4.training.training_gnn.gnn_evaluator import GNNEvaluator
from task4.training.training_gnn.graph import *
import torch

log = logging.getLogger(__name__)


class Coach():
    def __init__(self, game, nnet, config):
        self.game = game
        self.nnet = nnet
        self.config = config
        self.strategy = get_distribution_strategy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with self.strategy.scope():
            self.pnet = self.nnet.__class__(config).to(self.device) 
            self.evaluator = GNNEvaluator(self.nnet)
            self.mcts = MCTSAgent(self.game, self.config['exploration_coefficient'], self.config['num_MCTS_sims'], self.evaluator)
            self.trainExamplesHistory = []  
            self.skipFirstSelfPlay = False  
            self.winRatesRandom = []
            self.winRatesMCTS = []

        if self.config['saveResults']:
            with open(self.config['resultsFilePath'],"a",newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Iteration","won_random","lost_random","draw_random","winning rate","new_model", "won_mcts","lost_mcts","draw_mcts","winning rate"])


    def executeEpisode(self, opponent=None):
        trainExamples = []
        
        state = self.game.new_initial_state()
        self.curPlayer = state.current_player()
        episodeStep = 0

        while not state.is_terminal():
            episodeStep += 1
            temp = int(episodeStep <= self.config['tempThreshold'])
            
            self.curPlayer = state.current_player()

            if opponent and state.current_player() == 1: 
                policy_vector, action = opponent.get_action(state, temp)
            else:
                policy_vector, action = self.mcts.step(state, temp)
            
            trainExamples.append([state_to_graph_data(state), policy_vector, state.current_player(), None])
            
            state.apply_action(action)
        
        reward = state.rewards()[self.curPlayer]
        result = [(x[0], x[1], reward * ((-1) ** (x[2] != self.curPlayer))) for x in trainExamples]
        return result


            
            
        
    def learn(self):
        for i in range(1, self.config['numIterations'] + 1):
            log.info(f'Starting Iteration #{i} ...')
            log.info('Running Self Play')
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.config['maxlenOfQueue'])

                for _ in tqdm(range(int(self.config['numEpisodes'] * 0.5)), desc="Self Play"):
                    self.mcts = MCTSAgent(self.config['exploration_coefficient'], self.config['num_MCTS_sims'], self.evaluator)
                    iterationTrainExamples += self.executeEpisode()
                
                for _ in tqdm(range(int(self.config['numEpisodes'] * 0.5)), desc="Play Against Greedy Agent"):
                    self.mcts = MCTSAgent(self.config['exploration_coefficient'], self.config['num_MCTS_sims'], self.evaluator)
                    iterationTrainExamples += self.executeEpisode(opponent=self.greedy_agent)




                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.config['numItersForTrainExamplesHistory']:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            self.saveTrainExamples(i - 1)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.config['checkpoint'], filename='temp_checkpoint.weights.h5')
            self.pnet.load_checkpoint(folder=self.config['checkpoint'], filename='temp_checkpoint.weights.h5')

            previous_mcts = MCTSAgent(self.config['exploration_coefficient'], self.config['num_MCTS_sims'], self.evaluator)
            log.info('Training Neural Network')
            self.nnet.train(trainExamples)
            self.nnet.save_checkpoint(folder=self.config['checkpoint'], filename='temp_checkpoint.weights.h5')

            new_mcts = MCTSAgent(self.config['exploration_coefficient'], self.config['num_MCTS_sims'], self.evaluator) 

            log.info('Comparing old and new network')
            arena = Arena(previous_mcts, new_mcts, self.game)
            pwins, nwins, draws = arena.playGames(self.config['arenaCompare'])

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.config['updateThreshold']:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.config['checkpoint'], filename='temp_checkpoint.weights.h5')
                new_model = False
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.config['checkpoint'], filename=f'checkpoint_{i}.weights.h5')
                self.nnet.save_checkpoint(folder=self.config['checkpoint'], filename='best.weights.h5')
            
                new_model = True
                won, lost, draw, examples_random = arena.playGamesAgainstRandom(new_mcts, 40)
                won_m, lost_m, draw_m, examples_mcts = arena.playGamesAgainstMCTS(new_mcts, 40)
                winning_rate = round(won/(won+lost+draw),4)
                winning_rate_m = round(won_m/(won_m+lost_m+draw_m),4)
                log.info(f"Against Random - Won: {won}, Lost: {lost}, Draw: {draw}, Winning Rate: {winning_rate}")
                log.info(f"Against MCTS - Won: {won_m}, Lost: {lost_m}, Draw: {draw_m}, Winning Rate: {winning_rate_m}")

            if new_model and self.config['saveResults']:
                results_path = os.path.join(self.config['checkpoint'], self.config['resultsFilePath'])
                with open(results_path, "a", newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([i, won, lost, draw, winning_rate, new_model, won_m, lost_m, draw_m, winning_rate_m])

        return self.winRatesRandom, self.winRatesMCTS

    def getCheckpointFile(self, iteration):
        return f'checkpoint_{iteration}.h5'

    def saveTrainExamples(self, iteration):
        folder = self.config['checkpoint']
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f'checkpoint_{iteration}.examples')
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
            log.info(f"Saved Train Examples. \n")
            log.info(f'checkpoint_{iteration}.examples')
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.config['load_folder_file_0'], self.config['load_folder_file_1'])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info(f'Length of training data: {len(self.trainExamplesHistory)}')
            log.info('Loading done!')

            self.skipFirstSelfPlay = True