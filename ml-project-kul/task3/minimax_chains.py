import pyspiel

import numpy as np
import time
from chains_util import *
from auxiliaryMethods import *
from sys import getsizeof
class MinimaxChains:
    def __init__(self, game):
        params = game.get_parameters()
        self.game = game
        self.num_rows = params['num_rows']
        self.num_cols = params['num_cols']

        self.num_horizontal_lines = (self.num_rows + 1) * self.num_cols
        self.num_vertical_lines = self.num_rows * (self.num_cols + 1)

        self.transposition_table = {}

        self.name = "Minimax Chains"

    def create_combined_grid(self, action_tensor):
        """
        Create a grid representation of the game state from an action tensor.
        Args:
          action_tensor (list): List of zeros and ones representing taken actions.
        Returns:
          2D numpy array representing the grid.
        """
        # Initialize an empty grid with enough space for lines and nodes
        grid = np.zeros((2 * self.num_rows + 1, 2 * self.num_cols + 1))

        # Fill in horizontal lines
        for i in range(self.num_rows + 1):
            for j in range(self.num_cols):
                index = i * self.num_cols + j
                grid[2 * i, 2 * j + 1] = action_tensor[index]

        # Fill in vertical lines
        offset = (self.num_rows + 1) * self.num_cols
        for i in range(self.num_rows):
            for j in range(self.num_cols + 1):
                index = offset + i * (self.num_cols + 1) + j
                grid[2 * i + 1, 2 * j] = action_tensor[index]
        return grid

    @staticmethod
    def rotate_grid(grid, times=1):
        """Rotate the grid 90 degrees clockwise 'times' times."""
        grid_copy = np.copy(grid)
        for _ in range(times):
            grid_copy = np.rot90(grid_copy, -1)
        return grid_copy

    @staticmethod
    def reflect_grid(grid, axis=0):
        """Reflect the grid across an axis. Axis 0 is vertical, 1 is horizontal."""
        return np.flip(grid, axis=axis)

    def get_canonical_form(self, action_tensor):
        grid = self.create_combined_grid(action_tensor)
        # Check if square board
        if self.num_rows == self.num_cols:
            forms = [
                grid,
                self.rotate_grid(grid, 1),
                self.rotate_grid(grid, 2),
                self.rotate_grid(grid, 3),
                self.reflect_grid(grid, 0),
                self.reflect_grid(grid, 1),
                self.rotate_grid(self.reflect_grid(grid, 0), 1),
                self.rotate_grid(self.reflect_grid(grid, 0), 2),
                self.rotate_grid(self.reflect_grid(grid, 0), 3)
            ]
        else:
            forms = [
                grid,
                self.rotate_grid(grid, 2),
                self.reflect_grid(grid, 0),
                self.reflect_grid(grid, 1),
                self.rotate_grid(self.reflect_grid(grid, 0), 2),
                self.rotate_grid(self.reflect_grid(grid, 1), 2)
            ]

        # Pick the lexicographically smallest transformation
        canonical_form = min(forms, key=lambda x: x.tostring())
        return canonical_form

    def state_to_key_symmetries(self, state):
        # Using transposition tables with symmetries
        # Total keys: 212
        # Execution time: 0.19 seconds
        # Generate a key for the state using its canonical form and current player.
        action_string = state.dbn_string()
        canonical_tensor = self.get_canonical_form(action_string)
        return tuple(canonical_tensor.flatten())

    def _minimax(self, state, maximizing_player_id, num_rows, num_cols, score = [], action = 0, all_moves=[], all_indices=[], depth=0, iteration=0):
        if state.is_terminal():
            return state.player_return(maximizing_player_id), all_moves, all_indices

        score = updateScore(state, num_rows, num_cols, score, action)
        key = self.state_to_key_symmetries(state), state.current_player(), len(score)

        if key in self.transposition_table.keys():
            return self.transposition_table[key]




        rows = int(str(state.get_game())[35 : 36])
        cols = int(str(state.get_game())[24 : 25])

        player = state.current_player()
        found = 0
        if player == maximizing_player_id:
            dummy_port = 0
            chain_action, chain_action_available = protocol_moveonly(dummy_port, dummy_port, all_moves, all_indices, player + 1, int(str(state.get_game())[35 : 36]), int(str(state.get_game())[24 : 25]))
            chain_action = edge_to_spiel_action(chain_action, rows, cols)
            if chain_action_available:
                found = 1
                best_value = 1

        if found == 0:

            best_value = float('-inf') if player == maximizing_player_id else float('inf')

            for action in state.legal_actions():
                child_state = state.child(action)

                formatted_action = dic_dict[(num_rows, num_cols)][action]

                value, moves, indices = self._minimax(child_state, maximizing_player_id, num_rows, num_cols, score, action,all_moves + [formatted_action], all_indices + [player], depth + 1, iteration + 1)

                if player == maximizing_player_id:
                    if value > best_value:
                        best_value = value
                else:
                    if value < best_value:
                        best_value = value

        if found == 0:
            self.transposition_table[key] = (best_value, all_moves , all_indices + [player])
        return best_value, all_moves, all_indices + [player]


    def minimax_search(self, game,
                       state=None,
                       maximizing_player_id=None,
                       state_to_key=lambda state: state):
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

        start_time = time.time()


        game_info = game.get_type()
        params = game.get_parameters()
        num_rows = params['num_rows']
        num_cols = params['num_cols']

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
        v = self._minimax(
            state.clone(),
            maximizing_player_id=maximizing_player_id,
            num_rows=num_rows,
            num_cols=num_cols,
        )
        total_keys = len(self.transposition_table)
        execution_time = time.time() - start_time
        size_in_mb = getsizeof(self.transposition_table) / (1024 * 1024)
        print("Dictionary size: " + str(size_in_mb) + " MB")
        return v[0], total_keys, execution_time

dict_for_23 = {0 : ((0, 0), (0, 1)),
                           1 : ((0, 1), (0, 2)),
                           2 : ((0, 2), (0, 3)),
                           3 : ((1, 0), (1, 1)),
                           4 : ((1, 1), (1, 2)),
                           5 : ((1, 2), (1, 3)),
                           6 : ((2, 0), (2, 1)),
                           7 : ((2, 1), (2, 2)),
                           8 : ((2, 2), (2, 3)),
                           9 : ((0, 0), (1, 0)),
                           10 : ((0, 1), (1, 1)),
                           11 : ((0, 2), (1, 2)),
                           12 : ((0, 3), (1, 3)),
                           13 : ((1, 0), (2, 0)),
                           14 : ((1, 1), (2, 1)),
                           15 : ((1, 2), (2, 2)),
                           16 : ((1, 3), (2, 3))}

dict_for_22 = {0: ((0, 0), (0, 1)),
                           1 : ((0, 1), (0, 2)),
                           2 : ((1, 0), (1, 1)),
                           3 : ((1, 1), (1, 2)),
                           4 : ((2, 0), (2, 1)),
                           5 : ((2, 1), (2, 2)),
                           6 : ((0, 0), (1, 0)),
                           7 : ((0, 1), (1, 1)),
                           8 : ((0, 2), (1, 2)),
                           9: ((1,0), (2, 0)),
                              10: ((1, 1), (2, 1)),
                           11 : ((1, 2), (2, 2))}


dict_for_12 = {0 : ((0, 0), (0, 1)),
                           1 : ((0, 1), (0, 2)),
                           2 : ((1, 0), (1, 1)),
                           3 : ((1, 1), (1, 2)),
                           4 : ((0, 0), (1, 0)),
                           5 : ((0, 1), (1, 1)),
                           6 : ((0, 2), (1, 2))}

dict_for_11 = {0 : ((0, 0), (0, 1)),
               1 : ((1,0), (1, 1)),
               2 : ((0, 0), (1, 0)),
                3 : ((0, 1), (1, 1))}

dict_for_13 = {0 : ((0, 0), (0, 1)),
               1 : ((0, 1), (0, 2)),
               2 : ((0, 2), (0, 3)),
               3 : ((1,0), (1, 1)),
               4 : ((1, 1), (1, 2)),
                5 : ((1, 2), (1, 3)),
               6 : ((0, 0), (1, 0)),
                7 : ((0, 1), (1, 1)),
               8 : ((0, 2), (1, 2)),
                9: ((0, 3), (1, 3))}

dict_for_14 = {0 : ((0, 0), (0, 1)),
               1 : ((0, 1), (0, 2)),
                2 : ((0, 2), (0, 3)),
                3 : ((0, 3), (0, 4)),
                4 : ((1,0), (1, 1)),
                5 : ((1, 1), (1, 2)),
                6 : ((1, 2), (1, 3)),
                7 : ((1, 3), (1, 4)),
                8 : ((0, 0), (1, 0)),
                9 : ((0, 1), (1, 1)),
                10 : ((0, 2), (1, 2)),
                11 : ((0, 3), (1, 3)),
                12 : ((0, 4), (1, 4))}

dict_for_24 = {0 : ((0, 0), (0, 1)),
               1 : ((0, 1), (0, 2)),
                2 : ((0, 2), (0, 3)),
                3 : ((0, 3), (0, 4)),
                4 : ((1,0), (1, 1)),
                5 : ((1, 1), (1, 2)),
                6 : ((1, 2), (1, 3)),
                7 : ((1, 3), (1, 4)),
                8 : ((2, 0), (2, 1)),
                9 : ((2, 1), (2, 2)),
                10 : ((2, 2), (2, 3)),
                11 : ((2, 3), (2, 4)),
                12 : ((0, 0), (1, 0)),
                13 : ((0, 1), (1, 1)),
                14 : ((0, 2), (1, 2)),
                15 : ((0, 3), (1, 3)),
                16 : ((0, 4), (1, 4)),
                17 : ((1, 0), (2, 0)),
                18 : ((1, 1), (2, 1)),
                19 : ((1, 2), (2, 2)),
                20 : ((1, 3), (2, 3)),
                21 : ((1, 4), (2, 4))}

dict_for_33 = {0 : ((0, 0), (0, 1)),
               1 : ((0, 1), (0, 2)),
                2 : ((0,2), (0, 3)),
                3 : ((1,0), (1, 1)),
                4 : ((1, 1), (1, 2)),
                5 : ((1, 2), (1, 3)),
                6 : ((2, 0), (2, 1)),
                7 : ((2, 1), (2, 2)),
                8 : ((2, 2), (2, 3)),
                9 : ((3, 0), (3, 1)),
                10 : ((3, 1), (3, 2)),
                11 : ((3, 2), (3, 3)),
                12 : ((0, 0), (1, 0)),
                13 : ((0, 1), (1, 1)),
                14 : ((0, 2), (1, 2)),
                15 : ((0, 3), (1, 3)),
                16 : ((1, 0), (2, 0)),
               17 : ((1, 1), (2, 1)),
                18 : ((1, 2), (2, 2)),
                19 : ((1, 3), (2, 3)),
                20 : ((2, 0), (3, 0)),
                21 : ((2, 1), (3, 1)),
                22 : ((2, 2), (3, 2)),
                23 : ((2, 3), (3, 3))}



dic_dict = {
    (1,1) : dict_for_11,
    (1,2) : dict_for_12,
    (1,3) : dict_for_13,
    (1,4) : dict_for_14,
    (2,3) : dict_for_23,
    (2,2) : dict_for_22,
    (2,4) : dict_for_24,
    (3,3) : dict_for_33
}