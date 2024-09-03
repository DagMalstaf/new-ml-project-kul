import numpy as np
import pyspiel
import logging
from pathos.multiprocessing import ProcessingPool as Pool
from task4.agent.mcts_openspiel.SearchNode import SearchNode
from task4.agent.mcts_openspiel.evaluator_V1 import RandomRolloutEvaluator
from task4.agent.mcts_openspiel.smart_agent_for_nn import *
import random

game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})

centre_moves = [16, 74, 17, 75, 18, 76, 23, 77, 24, 82, 25, 83, 30, 84, 31, 85, 32, 90, 37, 91, 38, 92, 39, 93]
spokes_moves = [0, 57, 6, 62, 42, 105, 48, 110]

player_0_index_to_action = {}
for i in range(0, 112):
    state = game.new_initial_state()
    initial_tensor = state.observation_tensor()
    state.apply_action(i)
    final_tensor = state.observation_tensor()
    for j in range(576//3, len(initial_tensor)):
        if (initial_tensor[j] != final_tensor[j]):
            player_0_index_to_action[j] = i
    
player_1_index_to_action = {384 : 0}
for i in range(1, 112):
    state = game.new_initial_state()
    state.apply_action(0)
    initial_tensor = state.observation_tensor()
    state.apply_action(i)
    final_tensor = state.observation_tensor()
    for j in range(576//3, len(initial_tensor)):
        if (initial_tensor[j] != final_tensor[j]):
            player_1_index_to_action[j] = i

class MCTSAgent():
    def __init__(self, game, uct_c, max_simulations, evaluator, solve=True,
                 random_state=None, child_selection_fn=SearchNode.uct_value,
                 dirichlet_noise=None, verbose=False, dont_return_chance_node=False):
        #super().__init__()
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
        self.process_pool = Pool(self.num_processes)
        self.simulations_count = 0
        self.max_utility = 1
        self.min_utility = -1
    
    def __del__(self):
        self.close_pool()
    
    def close_pool(self):
        if self.process_pool:
            self.process_pool.close()
            self.process_pool.join()
            self.process_pool = None
  
    def parallel_mcts_search(self, state):
        cloned_state = state.clone()
        serialized_state = cloned_state.serialize()
        serialized_states = [serialized_state for _ in range(self.num_processes)]
        result_objs = self.process_pool.amap(self.mcts_search, serialized_states)
        result_objs.wait()
        results = result_objs.get()
        return self.aggregate_results(results)

    def mcts_search(self, serialized_state):
        state = self._game.deserialize_state(serialized_state)
        counter = 0
        root = SearchNode(None, state.current_player(), 1)
        for _ in range(self.max_simulations):
            counter += 1
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

    def aggregate_results(self, results):
        if not results:
            raise ValueError("No results to aggregate.")

        roots, counters = zip(*results)
        aggregated_root = SearchNode(None, roots[0].player, 1)
        action_map = {}
        total_count = sum(counters)

        detailed_info = {}


        for root in roots:
            for child in root.children:
                if child.action not in action_map:
                    action_map[child.action] = SearchNode(child.action, child.player, child.prior)
                    action_map[child.action].explore_count = child.explore_count
                    action_map[child.action].total_reward = child.total_reward
                    action_map[child.action].outcome = child.outcome

                    detailed_info[child.action] = [(child.explore_count, child.total_reward, child.outcome)]

                else:
                    existing_node = action_map[child.action]
                    existing_node.explore_count += child.explore_count
                    existing_node.total_reward += child.total_reward
                    if existing_node.outcome is not None and existing_node.outcome != child.outcome:
                        existing_node.outcome = None 
                    
                    detailed_info[child.action].append((child.explore_count, child.total_reward, child.outcome))

        aggregated_root.children = list(action_map.values())

        return aggregated_root, total_count

    def step_with_policy(self, state):
        root, count = self.parallel_mcts_search(state)
        self.simulations_count = count
        best = root.best_child()
    
        mcts_action = best.action
        policy = [(action, (1.0 if action == mcts_action else 0.0))
                  for action in state.legal_actions(state.current_player())]
        
        return policy, mcts_action

    def form_game_string(self, state):
        state_custom_string = "0" * 112
        state_custom_list = list(state_custom_string)
        for key in player_0_index_to_action.keys():
            tensor = state.observation_tensor()
            if tensor[key] == 1:
                state_custom_list[player_0_index_to_action[key]] = "1"

        for key in player_1_index_to_action.keys():
            tensor = state.observation_tensor()
            if tensor[key] == 1:
                state_custom_list[player_1_index_to_action[key]] = "2"

        state_custom_string = "".join(state_custom_list)
        return state_custom_string

    def get_spokes_move(self, state_custom_string, neg_moves):
        for i in range(0, len(spokes_moves)):
            if (state_custom_string[spokes_moves[i]] == '0' and spokes_moves[i] not in neg_moves):
                return spokes_moves[i], True
        return spokes_moves[i], False

    def get_centre_move(self, state_custom_string, neg_moves):
        for i in range(0, len(centre_moves)):
            if (state_custom_string[centre_moves[i]] == '0' and centre_moves[i] not in neg_moves):
                return centre_moves[i], True
        return centre_moves[i], False

    def get_chain_action_if_present(self, state_custom_string):
        all_moves = []
        all_indices = []
        for i in range(0, len(state_custom_string)):
            if (state_custom_string[i] == '1'):
                all_indices.append(1)
                action = i
                if (action < 56):
                    all_moves.append(((action//7, action%7), (action//7, action%7 + 1)))
                else:
                    all_moves.append((((action - 56)%7, (action - 56)//7), ((action - 56)%7 + 1, (action - 56)//7)))
            if (state_custom_string[i] == '2'):
                all_indices.append(2)
                action = i
                if (action < 56):
                    all_moves.append(((action//7, action%7), (action//7, action%7 + 1)))
                else:
                    all_moves.append((((action - 56)%7, (action - 56)//7), ((action - 56)%7 + 1, (action - 56)//7)))

        chain_action, chain_action_available = protocol_moveonly(80, 80, all_moves, all_indices)
        chain_action = edge_to_spiel_action(chain_action)
        return chain_action, chain_action_available

    def step(self, state):
        prob = random.random()
        if (prob > 0.8):
            action, action_available = self.get_spokes_move(state_custom_string, [])
            if (action_available == False):
                action, action_available = self.get_centre_move(state_custom_string, [])
        else:
            action, action_available = self.get_centre_move(state_custom_string, [])
            if (action_available == False):
                action, action_available = self.get_spokes_move(state_custom_string, [])
        state_custom_string = self.form_game_string(state)
        chain_action, chain_action_available = self.get_chain_action_if_present(state_custom_string)
        if (chain_action_available == True):
            return chain_action
        else:
            return self.step_with_policy(state)[1]
    
    def get_action(self,state, time):
        return self.step(state)


def get_agent_for_tournament(player_id):
    game = pyspiel.load_game("dots_and_boxes", {"num_rows": 7, "num_cols": 7})
    num_simulations = 25
    exploration_coefficient = 1.414
    evaluator = RandomRolloutEvaluator(5)

    agent = MCTSAgent(game, exploration_coefficient, num_simulations, evaluator)

    #logging.info(f"MCTSAgent OS V4 created for tournament with player ID: {player_id}")
    return agent
