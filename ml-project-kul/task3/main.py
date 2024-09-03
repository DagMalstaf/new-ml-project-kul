import pyspiel
from absl import app
from minimax_transposition import MinimaxTransposition
from minimax_symmetry import MinimaxSymmetry
from minimax_chains import MinimaxChains
from minimax_template import MinimaxTemplate
import time
import os
import gc

def main(_):
    games_list = pyspiel.registered_names()
    assert "dots_and_boxes" in games_list
    game_string = "dots_and_boxes(num_rows=2,num_cols=3)"

    print("Creating game: {}".format(game_string))
    game = pyspiel.load_game(game_string)

    #minimax_types = [MinimaxTemplate]
    minimax_types = [MinimaxTransposition, MinimaxSymmetry, MinimaxChains]

    for MS in minimax_types:
        print(f"Running Minimax Search with {MS.__name__}")
        gc.collect()
        ms = MS(game)
        first_time = time.time()
        value, total_keys, execution_time = ms.minimax_search(game)
        #value = ms.minimax_search(game)

        if value == 0:
            print("It's a draw")
        else:
            winning_player = 1 if value == 1 else 2
            print(f"Player {winning_player} wins.")
        execution_time = time.time() - first_time
        print(f"Total keys: {total_keys}")
        print(f"Execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    app.run(main)
