import os
import sys

package_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(package_directory)

from transposition_table import Transposition_Table
from chains_strategy import StrategyAdvisor

def _minimax(state,
            depth, 
            maximizing_player_id, 
            value_function, 
            cache:Transposition_Table, 
            strategy_advisor:StrategyAdvisor):
    """
    Implements a min-max algorithm

    Arguments:
      state: The current state node of the game.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN.

    Returns:
      The optimal value of the sub-game starting in state
    """
    if state.is_terminal():
        return state.player_return(maximizing_player_id)

    # if value_function is not None and depth <= 0 : 
    #     print("nice")
    #     return value_function(state, maximizing_player_id)

    val = cache.get(state)
    if val != None:
        return val
    
    player = state.current_player()
    if player == maximizing_player_id:
        selection = max
    else:
        selection = min

    possible_acitons = state.legal_actions()
    values_children = []
    for action in possible_acitons:
        child_state = state.child(action)
        value = _minimax(child_state, depth-1, maximizing_player_id, value_function, cache, strategy_advisor)
        values_children.append(value)
    
    minimax_value = selection(values_children)
    cache.set(state, minimax_value)

    return minimax_value

def _minimax_chains(state,
            depth, 
            maximizing_player_id, 
            value_function, 
            cache:Transposition_Table, 
            strategy_advisor:StrategyAdvisor):
    """
    Implements a min-max algorithm

    Arguments:
      state: The current state node of the game.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN.

    Returns:
      The optimal value of the sub-game starting in state
    """
    if state.is_terminal():
        return state.player_return(maximizing_player_id)
    
    # if value_function is not None and depth <= 0 : 
    #     return value_function(state, maximizing_player_id)

    val = cache.get(state)
    if val != None:
        return val

    player = state.current_player()
    if player == maximizing_player_id:
        selection = max
    else:
        selection = min

    possible_acitons = strategy_advisor.get_available_action(state)
    values_children = []
    for action in possible_acitons:

        child_state = state.clone()
        child_SA = strategy_advisor.clone()

        child_state.apply_action(action)
        child_SA.update_action(action, state.current_player())

        value = _minimax_chains(child_state, depth-1, maximizing_player_id, value_function, cache, child_SA)
        values_children.append(value)
    
    minimax_value = selection(values_children)
    cache.set(state, minimax_value)

    return minimax_value
