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
from task4.tuning.tuning_gnn.arena import Arena
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pnet = self.nnet.__class__(config).to(self.device) 
        self.evaluator = GNNEvaluator(self.nnet)
        self.mcts = MCTSAgent(self.config['exploration_coefficient'], self.config['num_MCTS_sims'], self.evaluator)
        self.trainExamplesHistory = []  
        self.skipFirstSelfPlay = True  
        self.winRatesRandom = []
        self.winRatesMCTS = []

    
    def learn(self):
        for i in range(1, self.config['numIterations'] + 1):
            
            log.info(f'Starting Iteration #{i} ...')
            
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            log.info('Training Neural Network')
            self.nnet.train(trainExamples)

            previous_mcts = MCTSAgent(self.config['exploration_coefficient'], self.config['num_MCTS_sims'], self.evaluator)
            new_mcts = MCTSAgent(self.config['exploration_coefficient'], self.config['num_MCTS_sims'], self.evaluator) 

            arena = Arena(previous_mcts, new_mcts, self.game)

            self.nnet.save_checkpoint(folder=self.config['checkpoint'], filename='best.weights.h5')
            won_m, lost_m, draw_m = arena.playGamesAgainstGreedy(new_mcts, 40, self.evaluator)
            #won_m, lost_m, draw_m = arena.playGamesAgainstMCTS(new_mcts, 60)

            winning_rate_m = round(won_m/(won_m+lost_m+draw_m),4)
            log.info(f"Against Greedy - Won: {won_m}, Lost: {lost_m}, Draw: {draw_m}, Winning Rate: {winning_rate_m}")
            
        return winning_rate_m

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