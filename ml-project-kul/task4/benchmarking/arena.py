import logging
log = logging.getLogger(__name__)
from tqdm import tqdm
import numpy as np
from open_spiel.python.algorithms.mcts import SearchNode
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator
from collections import deque
import pyspiel
import time


class Arena():
    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        

    def playGame(self, verbose=False):
        trainExamples = []

        players = [self.player1, self.player2]
        state = self.game.new_initial_state()
        curPlayer = state.current_player()
        episodeStep = 0

        while not state.is_terminal():
            episodeStep += 1
            temp = int(episodeStep <= 35)

            curPlayer = state.current_player()
          
            bot = players[curPlayer]
            policy_vector, action = bot.step(state, temp)
            
            trainExamples.append([np.array(state.observation_tensor()), policy_vector, state.current_player(), None])
            state.apply_action(action)
            
        reward = state.rewards()[curPlayer]
        trainExamples = [(x[0], x[1], reward * ((-1) ** (x[2] != curPlayer))) for x in trainExamples]
        return state.rewards(), trainExamples
    
    def playGameWithExamples(self):

        players = [self.player1, self.player2]
        state = self.game.new_initial_state()
        curPlayer = state.current_player()
        episodeStep = 0
        
        while not state.is_terminal():
            episodeStep += 1
            temp = int(episodeStep <= 35)
            
            curPlayer = state.current_player()

            bot = players[curPlayer]
            action = bot.step(state)

            state.apply_action(action)

        return state.rewards()

    def playGames(self, num, verbose=False):
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        trainExamples = deque([], maxlen=80000)
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult, gameExamples = self.playGame(verbose=verbose)
            trainExamples += gameExamples
            if gameResult[0] == 1.0:
                oneWon += 1
            elif gameResult[1] == 1.0:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult, gameExamples = self.playGame(verbose=verbose)
            trainExamples += gameExamples
            if gameResult[1] == 1.0:
                oneWon += 1
            elif gameResult[0] == 1.0:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws, trainExamples
    
    def playGamesAgainstRandom(self, player1, num, verbose=False):
        rp2 = UniformRandomBot(1,np.random, self.game)
        arena = Arena(player1, rp2, self.game)
        oneWon, twoWon, draws = 0,0,0
        
        for i in tqdm(range(num//2), desc="Playing_games_1"):
            reward = arena.playGameWithExamples()
            if reward[0] == 1.0:
                oneWon += 1
            elif reward[1] == 1.0:
                twoWon += 1
            else:
                draws += 1
        
        rp2 = UniformRandomBot(0,np.random, self.game)
        arena = Arena(rp2,player1, self.game)

        for _ in tqdm(range(num//2), desc="Playing_games_2"):
            reward = arena.playGameWithExamples()
            if reward[0] == 1.0:
                twoWon += 1
            elif reward[1] == 1.0:
                oneWon += 1
            else:
                draws += 1
        return oneWon, twoWon, draws
    

    def playGamesAgainstMCTS(self, player1, num, verbose=False):
        evaluator = RandomRolloutEvaluator(1, np.random)
        p2 = MCTSBot(self.game, 1, 10, evaluator)
        arena = Arena(player1, p2, self.game)
        oneWon, twoWon, draws = 0, 0, 0

        for _ in tqdm(range(num // 2), desc="Playing_games_1"):
            reward = arena.playGameWithExamples()
            if reward[0] == 1.0:
                oneWon += 1
            elif reward[1] == 1.0:
                twoWon += 1
            else:
                draws += 1

        arena = Arena(p2, player1, self.game)

        for _ in tqdm(range(num // 2), desc="Playing_games_2"):
            reward = arena.playGameWithExamples()
            if reward[0] == 1.0:
                twoWon += 1
            elif reward[1] == 1.0:
                oneWon += 1
            else:
                draws += 1
        
        return oneWon, twoWon, draws
    
    
    
################################################################
# BENCHMARK BOTS



class UniformRandomBot(pyspiel.Bot):
  def __init__(self, player_id, rng, game):
    pyspiel.Bot.__init__(self)
    self._player_id = player_id
    self._rng = rng
    self.game = game
    self.initial_state = self.game.new_initial_state()
    self.actions = self.initial_state.legal_actions()

  def restart_at(self, state):
    pass

  def player_id(self):
    return self._player_id

  def provides_policy(self):
    return True

  def step_with_policy(self, state):
    num_actions = state.num_distinct_actions()
    p = 1 / num_actions
    policy = [(action, p) for action in self.actions]
    legal_actions = state.legal_actions()
    action = self._rng.choice(legal_actions)
    return policy, action

  def step(self, state):
    return self.step_with_policy(state)[1]
  

class MCTSBot(pyspiel.Bot):

  def __init__(self,
               game,
               uct_c,
               max_simulations,
               evaluator,
               solve=True,
               random_state=None,
               child_selection_fn=SearchNode.uct_value,
               dirichlet_noise=None,
               verbose=False,
               dont_return_chance_node=False):
   
    pyspiel.Bot.__init__(self)
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
      raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("Game must have sequential turns.")

    self._game = game
    self.uct_c = uct_c
    self.max_simulations = max_simulations
    self.evaluator = evaluator
    self.verbose = verbose
    self.solve = solve
    self.max_utility = game.max_utility()
    self._dirichlet_noise = dirichlet_noise
    self._random_state = random_state or np.random.RandomState()
    self._child_selection_fn = child_selection_fn
    self.dont_return_chance_node = dont_return_chance_node

  def restart_at(self, state):
    pass

  def step_with_policy(self, state):
    root = self.mcts_search(state)
    best = root.best_child()
    mcts_action = best.action

    total_counts = sum([child.explore_count for child in root.children])
    policy = [(child.action,  child.explore_count/total_counts) for child in root.children]

    for i in range(state.num_distinct_actions()):
        if i not in [x[0] for x in policy]:
            policy.append((i,0.0))

    policy = sorted(policy, key=lambda x: x[0])
    
    policy = [x[1] for x in policy]

    return policy, mcts_action

  def step(self, state):
    return self.step_with_policy(state)[1]

  def _apply_tree_policy(self, root, state):
    visit_path = [root]
    working_state = state.clone()
    current_node = root
    while (not working_state.is_terminal() and
           current_node.explore_count > 0) or (
               working_state.is_chance_node() and self.dont_return_chance_node):
      if not current_node.children:
        legal_actions = self.evaluator.prior(working_state)
        if current_node is root and self._dirichlet_noise:
          epsilon, alpha = self._dirichlet_noise
          noise = self._random_state.dirichlet([alpha] * len(legal_actions))
          legal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                           for (a, p), n in zip(legal_actions, noise)]
        self._random_state.shuffle(legal_actions)
        player = working_state.current_player()
        current_node.children = [
            SearchNode(action, player, prior) for action, prior in legal_actions
        ]

      if working_state.is_chance_node():
        outcomes = working_state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = self._random_state.choice(action_list, p=prob_list)
        chosen_child = next(
            c for c in current_node.children if c.action == action)
      else:
        chosen_child = max(
            current_node.children,
            key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                c, current_node.explore_count, self.uct_c))

      working_state.apply_action(chosen_child.action)
      current_node = chosen_child
      visit_path.append(current_node)

    return visit_path, working_state

  def mcts_search(self, state):
    root = SearchNode(None, state.current_player(), 1)
    for _ in range(self.max_simulations):
      visit_path, working_state = self._apply_tree_policy(root, state)
      if working_state.is_terminal():
        returns = working_state.returns()
        visit_path[-1].outcome = returns
        solved = self.solve
      else:
        returns = self.evaluator.evaluate(working_state)
        solved = False

      while visit_path:
        decision_node_idx = -1
        while visit_path[decision_node_idx].player == pyspiel.PlayerId.CHANCE:
          decision_node_idx -= 1
        target_return = returns[visit_path[decision_node_idx].player]
        node = visit_path.pop()
        node.total_reward += target_return
        node.explore_count += 1

        if solved and node.children:
          player = node.children[0].player
          if player == pyspiel.PlayerId.CHANCE:
            outcome = node.children[0].outcome
            if (outcome is not None and
                all(np.array_equal(c.outcome, outcome) for c in node.children)):
              node.outcome = outcome
            else:
              solved = False
          else:
            best = None
            all_solved = True
            for child in node.children:
              if child.outcome is None:
                all_solved = False
              elif best is None or child.outcome[player] > best.outcome[player]:
                best = child
            if (best is not None and
                (all_solved or best.outcome[player] == self.max_utility)):
              node.outcome = best.outcome
            else:
              solved = False
      if root.outcome is not None:
        break

    return root

