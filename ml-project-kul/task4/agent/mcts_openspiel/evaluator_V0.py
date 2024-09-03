from open_spiel.python.algorithms.mcts import Evaluator

class SimpleEvaluator(Evaluator):
    def __init__(self):
        pass

    def evaluate(self, state):
        """Returns a simple evaluation of the given state."""
        return state.returns()

    def prior(self, state):
        """Returns a uniform prior probability over all legal actions."""
        actions = state.legal_actions()
        num_actions = len(actions)
        return [(action, 1/num_actions) for action in actions]  # Uniform probability

