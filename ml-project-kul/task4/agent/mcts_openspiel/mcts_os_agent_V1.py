import numpy as np
import pyspiel
import logging
from open_spiel.python.algorithms.mcts import SearchNode
from task4.agent.mcts_openspiel.evaluator_V1 import RandomRolloutEvaluator


class MCTSAgent(pyspiel.Bot):
    def __init__(self, game, uct_c, max_simulations, evaluator, solve=True,
                 random_state=None, child_selection_fn=SearchNode.uct_value,
                 dirichlet_noise=None, verbose=False, dont_return_chance_node=False):
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
        self._dirichlet_noise = dirichlet_noise
        self._random_state = random_state or np.random.RandomState()
        self._child_selection_fn = child_selection_fn
        self.dont_return_chance_node = dont_return_chance_node
        self.num_processes = 4
        self.process_pool = None
        self.simulations_count = 0
        self.max_utility = self._game.max_utility()

        
    def restart_at(self, state):
        pass

    def step_with_policy(self, state, temp):
        root, count = self.mcts_search(state)
        self.simulations_count = count
        best = root.best_child()
    
        mcts_action = best.action

        if temp == 0:
            max_index = mcts_action
            return [1.0 if i == max_index else 0.0 for i in range(state.num_distinct_actions())], mcts_action

        total_counts = sum([child.explore_count for child in root.children])
        policy = [(child.action,  child.explore_count/total_counts) for child in root.children]

        for i in range(state.num_distinct_actions()):
            if i not in [x[0] for x in policy]:
                policy.append((i,0.0))

        policy = sorted(policy, key=lambda x: x[0])
        
        policy = [x[1] for x in policy]
        
        return policy, mcts_action

    def get_action(self,state, time):
        return self.step(state)

    def step(self, state):
        return self.step_with_policy(state)

    def _apply_tree_policy(self, root, state):
        visit_path = [root]
        working_state = state.clone()
        current_node = root
        while (not working_state.is_terminal() and current_node.explore_count > 0) or ( working_state.is_chance_node() and self.dont_return_chance_node):
            if not current_node.children:
                legal_actions = self.evaluator.prior(working_state)
                if current_node is root and self._dirichlet_noise:
                    epsilon, alpha = self._dirichlet_noise
                    noise = self._random_state.dirichlet([alpha] * len(legal_actions))
                    legal_actions = [(a, (1 - epsilon) * p + epsilon * n) for (a, p), n in zip(legal_actions, noise)]
                
                self._random_state.shuffle(legal_actions)
                player = working_state.current_player()
                current_node.children = [ SearchNode(action, player, prior) for action, prior in legal_actions ]

            if working_state.is_chance_node():
                outcomes = working_state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = self._random_state.choice(action_list, p=prob_list)
                chosen_child = next(c for c in current_node.children if c.action == action)
            else:
                chosen_child = max( current_node.children, key=lambda c: self._child_selection_fn(c, current_node.explore_count, self.uct_c))

            working_state.apply_action(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)

        return visit_path, working_state

    def mcts_search(self, state):
        root = SearchNode(None, state.current_player(), 1)
        counter = 0
        for i in range(self.max_simulations):
            counter += 1
            self.simulations_count += 1
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
                        if (outcome is not None and all(np.array_equal(c.outcome, outcome) for c in node.children)):
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
                        if (best is not None and (all_solved or best.outcome[player] == self.max_utility)):
                            node.outcome = best.outcome
                        else:
                            solved = False
            if root.outcome is not None:
                break

        return root, counter
  
def get_agent_for_tournament(player_id):
    game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})
    num_simulations = 10
    exploration_coefficient = 1.414
    evaluator = RandomRolloutEvaluator(5)

    agent = MCTSAgent(game, exploration_coefficient, num_simulations, evaluator)

    logging.info(f"MCTSAgent OS V1 created for tournament with player ID: {player_id}")
    return agent


