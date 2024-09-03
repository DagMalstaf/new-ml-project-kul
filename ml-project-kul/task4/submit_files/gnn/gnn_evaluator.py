import numpy as np
from open_spiel.python.algorithms.mcts import Evaluator
from graph import *

class GNNEvaluator(Evaluator):
    def __init__(self, nnet):
        self.nnet = nnet

    def evaluate(self, state):
        _,v = self.nnet.predict(state)

        v_cpu = v.cpu().numpy() 
        return np.array([v_cpu, -v_cpu])

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            pi, _ = self.nnet.predict(state=state)

            pi_cpu = pi.cpu().numpy()  

            pi = edges_to_actions(state, pi_cpu.tolist())

            result = [(action, pi[action]) for action, val in enumerate(state.legal_actions_mask()) if val]
            
            return result
        

            