import os
import sys
import numpy as np

package_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(package_directory)

import symmetries as s
# from memory_profiler import profile
from util import vectors_to_dbn, dbn_to_vectors
import pyspiel

class Transposition_Table:
    def __init__(self, num_rows, num_cols):
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.symmhits = 0
        self.num_rows = num_rows
        self.num_cols = num_cols

    def get(self, state:pyspiel.DotsAndBoxesState, playerid):
        hashed_state = self._hash_state(state, playerid)
        if hashed_state in self.cache:
            self.hits += 1
            return self.cache[hashed_state]
        else:
            self.misses += 1
            return None

    def set(self, state:pyspiel.DotsAndBoxesState, playerid, value):
        hashed_state = self._hash_state(state, playerid)
        self.cache[hashed_state] = value

    def _hash_state(self, state:pyspiel.DotsAndBoxesState, playerid:int):
        # difference = int(players_points[0] - players_points[1])
        hash_string = state.dbn_string()
        # hash_string += '#'+ str(difference) 
        hash_string += '$' + str(playerid)
        return hash_string
    
    def get_hits(self):
        return self.hits
    
    def get_misses(self):
        return self.misses
    
    def get_cache(self) -> dict:
        return self.cache.copy()
    
    def get_symmhits(self):
        return self.symmhits
    
class TEmpty_Table(Transposition_Table):
    def get(self, state:pyspiel.DotsAndBoxesState, playerid):
        self.misses += 1
        return None

    def set(self, state, playerid, value):
        return

class TOptimised_Table(Transposition_Table):
    def get(self, state:pyspiel.DotsAndBoxesState, playerid):
        hashed_dbn = self._hash_state(state, playerid)
        # dbn, player_id = hashed_dbn.split('$')  
        
        # params = state.get_game().get_parameters()
        # num_rows = params['num_rows']
        # num_cols = params['num_cols']
        
        if hashed_dbn in self.cache:
            self.hits += 1
            return self.cache[hashed_dbn]

        # h_matrix, v_matrix = dbn_to_vectors(num_rows, num_cols, dbn)
        # horizontal_h_list, horizontal_v_list = np.flipud(h_matrix), np.flipud(v_matrix)
        
        # cache_result = vectors_to_dbn(num_rows, num_cols, horizontal_h_list, horizontal_v_list)
        # cache_result = cache_result + player_id
        # if cache_result in self.cache:
        #     self.symmhits += 1
        #     return self.cache[cache_result]
        
        # vertical_h_list, vertical_v_list = np.fliplr(h_matrix), np.fliplr(v_matrix)
        # cache_result = vectors_to_dbn(num_rows, num_cols, vertical_h_list, vertical_v_list)
        # del vertical_h_list, vertical_v_list
        
        # cache_result = cache_result + player_id
        # if cache_result in self.cache:
        #     self.symmhits += 1
        #     return self.cache[cache_result]

        # if num_rows == num_cols:

        #     rot3_h_list, rot3_v_list = np.rot90(v_matrix, 3), np.rot90(h_matrix, 3)
        #     cache_result = vectors_to_dbn(num_rows, num_cols, rot3_h_list, rot3_v_list)
        #     del rot3_h_list, rot3_v_list

        #     cache_result = cache_result + player_id
        #     if cache_result in self.cache:
        #         self.symmhits += 1
        #         return self.cache[cache_result]
            
        #     rot1_h_list, rot1_v_list = np.rot90(v_matrix, 1), np.rot90(h_matrix, 1)
        #     cache_result = vectors_to_dbn(num_rows, num_cols, rot1_h_list, rot1_v_list)
        #     del rot1_h_list, rot1_v_list

        #     cache_result = cache_result + player_id
        #     if cache_result in self.cache:
        #         self.symmhits += 1
        #         return self.cache[cache_result]
            
        #     rotv_h_list, roth_v_list = np.rot90(horizontal_v_list, 3), np.rot90(horizontal_h_list, 3)
        #     cache_result = vectors_to_dbn(num_rows, num_cols, rotv_h_list, roth_v_list)
        #     del horizontal_v_list, horizontal_h_list, rotv_h_list, roth_v_list

        #     cache_result = cache_result + points
        #     if cache_result in self.cache:
        #         self.symmhits += 1
        #         return self.cache[cache_result]
            # else:
            #     self.misses += 1
            #     return None

        # hv_h_list, hv_v_list = np.fliplr(horizontal_h_list), np.fliplr(horizontal_v_list)
        # cache_result =  vectors_to_dbn(num_rows, num_cols, hv_h_list, hv_v_list)
        # del hv_h_list, hv_v_list

        # cache_result = cache_result + points
        # if cache_result in self.cache:
        #     self.symmhits += 1
        #     return self.cache[cache_result]

        else:
            self.misses += 1
            return None


class Transposition_Table_Chains:
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, hashed_state):
        if hashed_state in self.cache:
            self.hits += 1
            return self.cache[hashed_state]
        else:
            self.misses += 1
            return None

    def set(self, hashed_state, value):
        self.cache[hashed_state] = value
    
    def get_hits(self):
        return self.hits
    
    def get_misses(self):
        return self.misses
    
    def get_cache(self) -> dict:
        return self.cache.copy()