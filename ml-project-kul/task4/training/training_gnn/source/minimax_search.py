import os
import sys
import pyspiel

package_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(package_directory)

from transposition_table import Transposition_Table
from chains_strategy import StrategyAdvisor
from minimax import _minimax_chains, _minimax


def minimax_search(game,
                   strategy_advisor:StrategyAdvisor,
                   transposition_table: Transposition_Table,
                   maximum_depth=10,
                   state=None,
                   value_function=None,
                   maximizing_player_id=None):
    """Solves deterministic, 2-players, perfect-information 0-sum game.

    For small games only! Please use keyword arguments for optional arguments.

    Arguments:
      game: The game to analyze, as returned by `load_game`.
      state: The state to run from.  If none is specified, then the initial state is assumed.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN. The default (None) will suppose the player at the root to be
        the MAX player.

    Returns:
      The value of the game for the maximizing player when both player play optimally.
    """
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

    value = -float("inf")
    for action in state.legal_actions():
        child_state = state.clone()
        child_state.apply_action(action)

        child_value = _minimax(
                        child_state,
                        -1,
                        value_function=value_function,
                        maximizing_player_id=maximizing_player_id,
                        cache=transposition_table,
                        strategy_advisor=None)
        
        if child_value > value:
            value = child_value
            best_action = action  

    return value, best_action

def minimax_chains_search(game,
                   strategy_advisor:StrategyAdvisor,
                   transposition_table: Transposition_Table,
                   maximum_depth=10,
                   state=None,
                   value_function=None,
                   maximizing_player_id=None):
    """Solves deterministic, 2-players, perfect-information 0-sum game.

    For small games only! Please use keyword arguments for optional arguments.

    Arguments:
      game: The game to analyze, as returned by `load_game`.
      state: The state to run from.  If none is specified, then the initial state is assumed.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN. The default (None) will suppose the player at the root to be
        the MAX player.

    Returns:
      The value of the game for the maximizing player when both player play optimally.
    """
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

    possible_actions = strategy_advisor.get_available_action(state)

    value = -float("inf")
    for action in possible_actions:
        child_state = state.clone()
        child_SA = strategy_advisor.clone()

        child_state.apply_action(action)
        child_SA.update_action(action, state.current_player())

        child_value = _minimax_chains(
                        child_state,
                        -1,
                        value_function=value_function,
                        maximizing_player_id=maximizing_player_id,
                        cache=transposition_table,
                        strategy_advisor=child_SA.clone())
        
        if child_value > value:
            value = child_value
            best_action = action  

    return value, best_action

