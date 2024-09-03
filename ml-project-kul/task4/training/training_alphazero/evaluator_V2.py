import numpy as np
import pyspiel
from open_spiel.python.utils import lru_cache
import logging
import tensorflow as tf

class Evaluator(object):
  def evaluate(self, state):
    raise NotImplementedError

  def prior(self, state):
    raise NotImplementedError

class AlphaZeroEvaluator(Evaluator):
  def __init__(self, game, model, cache_size=2**16):
    if game.num_players() != 2:
      raise ValueError("Game must be for two players.")
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
      raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("Game must have sequential turns.")

    self._model = model
    self._cache = lru_cache.LRUCache(cache_size)

  def cache_info(self):
    return self._cache.info()

  def clear_cache(self):
    self._cache.clear()

  def _inference(self, state):
    obs = np.expand_dims(state.observation_tensor(), 0)
    mask = np.expand_dims(state.legal_actions_mask(), 0)

    cache_key = obs.tobytes() + mask.tobytes()

    try:
      result = self._cache.make(
          cache_key, lambda: self._model.predict(state))
    except tf.errors.OutOfRangeError:
                pass
    
    return result[0], result[1]

  def evaluate(self, state):
    result = self._inference(state)
    value = result[1]
    return np.array([value, -value])

  def prior(self, state):
    if state.is_chance_node():
      return state.chance_outcomes()
    else:
      policy, _ = self._inference(state)
      return [(action, policy[action]) for action in state.legal_actions()]