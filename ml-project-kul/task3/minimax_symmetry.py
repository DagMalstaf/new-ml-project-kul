import pyspiel
from absl import app
from auxiliaryMethods import *
from time import time
import numpy as np
from sys import getsizeof
class MinimaxSymmetry:
    def __init__(self, game):
        self.game = game
        params = game.get_parameters()
        self.num_rows = params['num_rows']
        self.num_cols = params['num_cols']
        self.name = "Minimax Symmetry New"

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



    def _minimax(self, state, maximizing_player_id, transpTable: dict, num_rows, num_cols, score = [], action = 0):
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

        # Update the list of scored cells after the last action
        score = updateScore(state, num_rows, num_cols, score, action)
        key = self.state_to_key_symmetries(state), state.current_player(), len(score)
        if key in transpTable.keys():
            return transpTable[key]
        else:
            player = state.current_player()
            if player == maximizing_player_id:
                selection = max
            else:
                selection = min
            values_children = [
                self._minimax(state.child(action), maximizing_player_id, transpTable, num_rows, num_cols, score, action)
                for
                action in state.legal_actions()]

            # Store the found value.
            result = selection(values_children)
            transpTable[key] = result

        return result


    def minimax_search(self, game,
                       state=None,
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

        params = game.get_parameters()
        num_rows = params['num_rows']
        num_cols = params['num_cols']
        transpTable = dict()
        start_time = time()


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
            transpTable=transpTable,
            num_rows=num_rows,
            num_cols=num_cols,
        )
        total_keys = len(transpTable)
        execution_time = time() - start_time
        size_in_mb = getsizeof(transpTable) / (1024 * 1024)
        print("Dictionary size: " + str(size_in_mb) + " MB")
        return v, total_keys, execution_time
