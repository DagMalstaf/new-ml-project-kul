import os
import sys
import pyspiel
import numpy as np

package_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(package_directory)

from transposition_table import TOptimised_Table, Transposition_Table_Chains
from chains_strategy import StrategyAdvisor
from evaluators import eval_maximize_difference

import pstats
import cProfile
# from memory_profiler import profile

def _alpha_beta(state, depth, alpha, beta, value_function,
                maximizing_player_id, 
                cache:TOptimised_Table,
                cache_chains:Transposition_Table_Chains, 
                SA:StrategyAdvisor):
    
    if state.is_terminal():
        # print(f"State is Terminal: {state} on depth {depth}")
        return state.player_return(maximizing_player_id)

    # terminal OF aantal boxes van ene speler is al meer dan de heflt. 
    # if (board.scores[0] > board.columns * board.rows / 2) {
    #     return 1;
    # }
    # if (board.scores[1] > board.columns * board.rows / 2) {
    #     return -1;
    # }

    if depth <= 0:
        # print(f"Depth bound reached:") 
        # print(f"{state} on depth {depth}")
        return value_function(state, maximizing_player_id, SA)
    
    val = cache.get(state, state.current_player())
    if val:
        # print(f"State was already in cache.")
        # print(state)
        # print(f"cache at depth {depth} utilised")
        return val
    
    possible_actions = SA.get_available_action(state, cache_chains, maximizing_player_id)
    # print(f"BEGIN on depth {depth}")
    # print(f'possible actions : {possible_actions}')

    player = state.current_player()
    if player == maximizing_player_id:
        value = -float("inf")
        for action in possible_actions:
            child_state = state.clone()
            child_SA = SA.clone()

            child_state.apply_action(action)
            child_SA.update_action(action, state.current_player())

            # print("maximum player trying to consider the following action.")
            # print(f"action: {SA.get_tabular_form(action)}")
        
            child_value = _alpha_beta(child_state, depth - 1, alpha, beta,
                                        value_function, maximizing_player_id, cache, cache_chains, child_SA)

            # print(f"value of child of maximum player with action : {child_value} and {SA.get_tabular_form(action)}, action: {action}")

            if child_value > value:
                value = child_value
            alpha = max(alpha, value)
            if alpha >= beta:
                # print(f"beta cut-off")
                break  # beta cut-off
        # print(f"END on depth {depth}")
        # transpostion table
        cache.set(state, state.current_player(), value)
        return value
    
    else:
        value = float("inf")
        for action in possible_actions:
            child_state = state.clone()
            child_SA = SA.clone()

            child_state.apply_action(action)
            child_SA.update_action(action, state.current_player())
            
            # print("minimum player trying to consider the following action.")
            # print(f"action: {SA.get_tabular_form(action)}")
        
            child_value  = _alpha_beta(child_state, depth - 1, alpha, beta,
                                        value_function, maximizing_player_id, cache, cache_chains, child_SA)
            
            # print(f"value of child of minimum player with action : {child_value} and {SA.get_tabular_form(action)} and action {action}")

            if child_value < value:
                value = child_value
            beta = min(beta, value)
            if alpha >= beta:
                # print(f"alpha cut-off")
                break  # alpha cut-off

        # print(f"END on depth {depth}")           
        # transpostion table
        cache.set(state, state.current_player(), value)
        return value
# @profile
def minimax_alphabeta_search(game,
                            transposition_table:TOptimised_Table,
                            transposition_table_chains:Transposition_Table_Chains,
                            strategy_advisor:StrategyAdvisor,
                            state=None,
                            value_function=None,
                            maximum_depth=10,
                            maximizing_player_id=None):
    """ functie met alles der op en der aan (van optimisaties) """

    game_info = game.get_type()

    if game.num_players() != 2:
        raise ValueError("Game must be a 2-player game")
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("The game must be a Deterministic one, not {}".format(
        game.chance_mode))
    if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
        raise ValueError(
        "The game must be a perfect information one, not {}".format(
            game.information))
    if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("The game must be turn-based, not {}".format(
        game.dynamics))
    if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
        raise ValueError("The game must be 0-sum, not {}".format(game.utility))

    if state is None:
        state = game.new_initial_state()
    if maximizing_player_id is None:
        maximizing_player_id = state.current_player()
    
    # print("intresting")
    # print(strategy_advisor.chains)
    
    possible_actions = strategy_advisor.get_available_action(state, transposition_table_chains, maximizing_player_id)
    
    # print("possible_actions")
    # print(possible_actions)

    value = -float("inf")
    for action in possible_actions:
        child_state = state.clone()
        child_SA = strategy_advisor.clone()

        child_state.apply_action(action)
        child_SA.update_action(action, state.current_player())
        # print("BEGIN")
        # print(f"action : {strategy_advisor.get_tabular_form(action)} or {action}")
        
        child_value = _alpha_beta(
                        child_state,
                        maximum_depth-1,
                        alpha=-float("inf"),
                        beta=float("inf"),
                        value_function=value_function,
                        maximizing_player_id=maximizing_player_id,
                        cache=transposition_table,
                        cache_chains = transposition_table_chains,
                        SA=child_SA.clone())
        
        
        # print(f"child_value : {child_value}")
        # print("END")

        # TODO: bij half haerted ding, als zelfde waarde is om hafl haerted en gwn fill box
        # dan fill box ?
        # want dan kiest het de eerste actie ...
        # wegens die > ipv >= 
        # als >= dan altijd tweede actie. 

        if child_value > value:
            value = child_value
            best_action = action  

    return value, best_action

def main():
    num_rows = 7
    num_cols = 7
    game_string = f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols})"

    # print("Creating game: {}".format(game_string))
    game = pyspiel.load_game(game_string)
    state = game.new_initial_state()
    SA = StrategyAdvisor(num_rows, num_cols)
    # print(SA.chains)
    max_allowed_depth = 1000

    maximizing_player_id = state.current_player()

    TT = TOptimised_Table(num_rows, num_cols)
    TTC = Transposition_Table_Chains()

    value, best_action = minimax_alphabeta_search(game=game,
                                        transposition_table=TT, 
                                        transposition_table_chains=TTC,
                                        strategy_advisor=SA,
                                        maximum_depth=max_allowed_depth,
                                        # TODO: can be changed to eval_chain
                                        value_function=eval_maximize_difference,
                                        state=state.clone())

    # print("next recommended action: ")
    # print(SA.get_tabular_form(best_action))  

    # print("the minimax value")
    # print(value)  

    # if value > 0 :
    #     print(f"In the simulation, Player {maximizing_player_id + 1} wins.")
    # elif value < 0 : 
    #     print(f"In the simulation, Player {maximizing_player_id} wins.")
    # else : 
    #     print("In the simulation, It's a draw")

    # print("Applying the recommended action to the state")
    # state.apply_action(best_action, )
    # print(state)


if __name__ == "__main__":
    # with cProfile.Profile() as profile: 
    main() 
    
    # results = pstats.Stats(profile)
    # results.sort_stats(pstats.SortKey.TIME)
    # results.print_stats()
    # results.dump_stats("results.prof")



# def _alpha_beta(state, depth, alpha, beta, value_function,
#                 maximizing_player_id, 
#                 cache:TOptimised_Table,
#                 cache_chains:Transposition_Table_Chains, 
#                 SA:StrategyAdvisor):
    
#     if state.is_terminal():
#         print(f"State is Terminal: {state} on depth {depth}")
#         return state.player_return(maximizing_player_id)

#     # terminal OF aantal boxes van ene speler is al meer dan de heflt. 
#     # if (board.scores[0] > board.columns * board.rows / 2) {
#     #     return 1;
#     # }
#     # if (board.scores[1] > board.columns * board.rows / 2) {
#     #     return -1;
#     # }

#     if depth <= 0:
#         print(f"Depth bound reached:") 
#         print(f"{state} on depth {depth}")
#         return value_function(state, maximizing_player_id)


#     # tt_entry = cache.get(state, SA.p_)
#     # if tt_entry is not None and tt_entry.depth >= depth:
#     #     if tt_entry.flag == TTEntry.EXACT:
#     #         return tt_entry.value
#     #     elif tt_entry.flag == TTEntry.LOWERBOUND:
#     #         alpha = max(alpha, tt_entry.value)
#     #     elif tt_entry.flag == TTEntry.UPPERBOUND:
#     #         beta = min(beta, tt_entry.value)
        
#     #     if alpha >= beta:
#     #         return tt_entry.value


#     possible_actions = SA.get_available_action(state, cache_chains, maximizing_player_id)
#     print(f"BEGIN on depth {depth}")
#     print(f'possible actions : {possible_actions}')

#     player = state.current_player()
#     if player == maximizing_player_id:
#         value = -float("inf")
#         for action in possible_actions:
#             child_state = state.clone()
#             child_SA = SA.clone()

#             child_state.apply_action(action)
#             child_SA.update_action(action, state.current_player())

#             print("maximum player trying to consider the following action.")
#             print(f"action: {SA.get_tabular_form(action)}")
        
#             child_value = _alpha_beta(child_state, depth - 1, alpha, beta,
#                                         value_function, maximizing_player_id, cache, cache_chains, child_SA)

#             print(f"value of child of maximum player with action : {child_value} and {SA.get_tabular_form(action)}, action: {action}")

#             if child_value > value:
#                 value = child_value

#             alpha = max(alpha, value)
#             if alpha >= beta:
#                 break  # beta cut-off
#         print(f"END on depth {depth}")
#         cache.set(state, SA.p_, value, alpha, beta, depth, cache)
#         return value
    
#     else:
#         value = float("inf")
#         for action in possible_actions:
#             child_state = state.clone()
#             child_SA = SA.clone()

#             child_state.apply_action(action)
#             child_SA.update_action(action, state.current_player())
            
#             print("minimum player trying to consider the following action.")
#             print(f"action: {SA.get_tabular_form(action)}")
        
#             child_value  = _alpha_beta(child_state, depth - 1, alpha, beta,
#                                         value_function, maximizing_player_id, cache, cache_chains, child_SA)
            
#             print(f"value of child of minimum player with action : {child_value} and {SA.get_tabular_form(action)} and action {action}")

#             if child_value < value:
#                 value = child_value
#             beta = min(beta, value)
#             if alpha >= beta:
#                 break  # alpha cut-off

#         print(f"END on depth {depth}")
#         cache.set(state, SA.p_, value, alpha, beta, depth, cache)   
#         return value
