import os
import sys
import math

parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

import random as r
import numpy as np
import time as t
from transposition_table import Transposition_Table_Chains
from util import CellOrientation, Directions, vectors_to_dbn
from evaluators import game_state_tensor, pvector_for_tensor, box_state_for_pvector
# type Alias
Action = int

# source : https://www.math.ucla.edu/~tom/Games/dots&boxes.html

class StrategyAdvisor : 
    def __init__(self, rows, cols):
        self.num_rows = rows
        self.num_cols = cols

        self.h_ = [[0] * cols for _ in range(rows + 1)]
        self.v_ = [[0] * (cols + 1) for _ in range(rows)]
        self.p_ = [0, 0]
        self.cells = [[0] * cols for _ in range(rows)]
        self.chains = {"half_open": [], "closed": [], "count": 0 }

    def clone(self):
        cloned_advisor = StrategyAdvisor(self.num_rows, self.num_cols)
        cloned_advisor.h_ = [row[:] for row in self.h_]
        cloned_advisor.v_ = [row[:] for row in self.v_]
        cloned_advisor.p_ = [self.p_[0], self.p_[1]]
        cloned_advisor.cells = [row[:] for row in self.cells]
        return cloned_advisor

    def get_tabular_form(self, action):
        maxh = (self.num_rows + 1) * self.num_cols
        if (action < maxh) : 
            # Horizontal
            orien = CellOrientation.HORIZONTAL
            row = math.floor(action / self.num_cols)
            col = action % self.num_cols
        else : 
            # Vertical
            action -= maxh
            orien = CellOrientation.VERTICAL
            row = math.floor(action / (self.num_cols + 1))
            col = action % (self.num_cols + 1)

        return orien, row, col
        
    def update_action(self, action, current_player):
        orien, i, j = self.get_tabular_form(action)
        if orien.value == CellOrientation.HORIZONTAL.value : 
            self.set_h_edge(i,j, current_player)
        else : 
            self.set_v_edge(i,j, current_player)         

    def update_edge(self, orien:CellOrientation, i, j, current_player):
        if orien.value == CellOrientation.HORIZONTAL.value : 
            self.set_h_edge(i,j, current_player)
        else : 
            self.set_v_edge(i,j, current_player)

    def set_h_edge(self, x, y, current_player):
        self.h_[x][y] = 1
        if x > 0:
            self.cells[x - 1][y] += 1
            if self.cells[x - 1][y] == 4: 
                self.p_[current_player] += 1

        if x < self.num_rows:
            self.cells[x][y] += 1
            if self.cells[x][y] == 4: 
                self.p_[current_player] += 1


    def set_v_edge(self, x, y, current_player):
        self.v_[x][y] = 1
        if y > 0:
            self.cells[x][y - 1] += 1
            if self.cells[x][y - 1] == 4: 
                self.p_[current_player] += 1

        if y < self.num_cols:
            self.cells[x][y] += 1
            if self.cells[x][y] == 4: 
                self.p_[current_player] += 1

    def action_id(self, orientation_:CellOrientation, row_:int, col_:int):
        """
        - Translate the row, column and orientation of the edge to an Action. 
        """

        action = 0
        maxh  = (self.num_rows + 1) * self.num_cols
        if (orientation_.value == CellOrientation.HORIZONTAL.value):
            action = row_ * self.num_cols + col_
        elif (orientation_.value == CellOrientation.VERTICAL.value):
            action = maxh + row_ * (self.num_cols + 1) + col_
        else : 
            raise ValueError("The orientation_ parameter must be of type CellOrientation")

        return action

    def safe3(self):
        """
        - Returns an actions that capture a singleton or a doubleton.
        """
        for row in range(0,self.num_rows):
            for col in range(0,self.num_cols):
                if self.cells[row][col] == 3:
                    if self.v_[row][col] < 1:
                        if col == 0 or self.cells[row][col-1] != 2 :
                            action = self.action_id(CellOrientation.VERTICAL, row, col)
                            return [action]
                        
                    elif self.h_[row][col] < 1:
                        if row == 0 or self.cells[row-1][col] != 2:
                            action = self.action_id(CellOrientation.HORIZONTAL, row, col)
                            return [action]

                    elif self.v_[row][col+1] < 1:
                        if col == self.num_cols-1 or self.cells[row][col+1] != 2:
                            action = self.action_id(CellOrientation.VERTICAL, row, col+1)
                            return [action]
                    else :
                        if row == self.num_rows-1 or self.cells[row+1][col] != 2:
                            action = self.action_id(CellOrientation.HORIZONTAL, row+1, col)
                            return [action]
        return None
    
    def find_all_chains(self):
        self.chains = {"half_open": [], "closed": [], "count": 0}
        points = self.side3()
        while points:
            u, v = points.pop(0)
            count = 0
            loop = False
            cellshist = []
            dirhist = []
            cellshist.append((u, v))
            count, loop = self.incount(0, u, v, count, loop, cellshist, dirhist)
            data_chain = {
                    "directions": dirhist,
                    "chain": cellshist,
                    "count" : count
                }
            
            if loop:
                self.chains["closed"].append(data_chain)
            else:
                self.chains["half_open"].append(data_chain)
            
            points = list(filter(lambda p: p not in cellshist, points))

            # We say a chain is long if it contains at least three boxes.
            # FIXME: It must be pointed out that in this rule, loops do not count as long chains. 
            if count >= 3:
                self.chains["count"] += 1 
    def side3(self):
        side3_list = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if self.cells[i][j] == 3:
                    side3_list.append((i,j)) 
        return side3_list
    
    def amount_of_capturable_boxes(self):
        """aantal boxes in een chain opgeteld over alle chains."""
        amnt = 0
        for closed_chain in self.chains["closed"]:
            amnt += closed_chain["count"]
        
        for half_open_chain in self.chains["half_open"]:
            amnt += half_open_chain["count"]
        return amnt

    def amount_of_boxes_captured(self): 
        """aantal boxes die al 4 edges hebben."""
        # count = 0
        # for i in range(self.num_rows) :
        #     for j in range(self.num_cols):
        #         if self.cells[i][j] == 4:
        #             count += 1 
        # return count
        return sum(self.p_)
    
    def unsafe3(self):
        amount_of_closed_chains     = len(self.chains["closed"])
        amount_of_half_open_chains  = len(self.chains["half_open"])
        amount_of_capturable_boxes  = self.amount_of_capturable_boxes()
        boxes_already_captured      = self.amount_of_boxes_captured()
        amount_of_boxes             = self.num_rows * self.num_cols

        # check if there are closed chains first. 
        if self.chains["closed"]:
            if (amount_of_closed_chains == 1 and 
                amount_of_half_open_chains == 0 and 
                amount_of_capturable_boxes == 4 and 
                # opt einde van de game, geen half haerted handout:
                amount_of_boxes - boxes_already_captured != amount_of_capturable_boxes):
                
                chain_data = self.chains["closed"][0]
                dhh_action = self.get_double_half_haerted(
                    chain_data["chain"],
                    chain_data["directions"])
                fb_action = self.get_fill_box(
                    chain_data["chain"],
                    chain_data["directions"])
                
                return [fb_action, dhh_action]
            
            else :
                chain_data = self.chains["closed"][0]
                fb_action = self.get_fill_box(
                    chain_data["chain"],
                    chain_data["directions"])
                return [fb_action]
          
        if self.chains["half_open"]:
            if (amount_of_closed_chains == 0 and 
                amount_of_half_open_chains == 1 and 
                amount_of_capturable_boxes == 2 and 
                # opt einde van de game, geen half haerted handout:
                amount_of_boxes - boxes_already_captured != amount_of_capturable_boxes):

                chain_data = self.chains["half_open"][0]
                shh_action = self.get_single_half_haerted(
                    chain_data["chain"],
                    chain_data["directions"])
                fb_action = self.get_fill_box(
                    chain_data["chain"],
                    chain_data["directions"])
                
                return [fb_action, shh_action]
            
            else :
                chain_data = self.chains["half_open"][0]
                fb_action = self.get_fill_box(
                    chain_data["chain"],
                    chain_data["directions"])
                return [fb_action]
        return None

    def get_single_half_haerted(self, half_open_chain: list[(int,int)], directions:list[Directions]):
        assert len(half_open_chain) == 2
        assert len(directions) == 2

        u1, v1 = half_open_chain[0]
        u2, v2 = half_open_chain[1]

        if self.cells[u1][v1] == 3:
            nu, nv = u2, v2
            nd = directions[1]
        elif self.cells[u2][v2] == 3: 
            nu, nv = u1, v1
            nd = directions[0]
        else :
            assert ValueError("Error in get_single_half_haerted")

        if nd.value == Directions.DOWN.value:
            action = self.action_id(CellOrientation.HORIZONTAL, nu+1, nv)
            return action
        elif nd.value == Directions.UP.value:
            action = self.action_id(CellOrientation.HORIZONTAL, nu, nv)
            return action
        elif nd.value == Directions.LEFT.value:
            action = self.action_id(CellOrientation.VERTICAL, nu , nv)
            return action
        else : 
            action = self.action_id(CellOrientation.VERTICAL, nu, nv+1)
            return action

    def get_double_half_haerted(self, closed_chain:list[(int,int)], directions:list[Directions]):
        assert len(closed_chain) == 4
        assert len(directions) == 3

        u, v = closed_chain[1]
        direction = directions[1]

        if direction.value == Directions.DOWN.value:
            action = self.action_id(CellOrientation.HORIZONTAL, u+1, v)
            return action
        elif direction.value == Directions.UP.value:
            action = self.action_id(CellOrientation.HORIZONTAL, u-1, v)
            return action
        elif direction.value == Directions.LEFT.value:
            action = self.action_id(CellOrientation.VERTICAL, u, v)
            return action
        else : 
            action = self.action_id(CellOrientation.VERTICAL, u, v+1)
            return action

    def get_fill_box(self, closed_chain:list[(int,int)], directions:list[Directions]):
        assert len(closed_chain) >= 1
        assert len(directions) >= 1

        counter = 0
        for u, v in closed_chain :
            if self.cells[u][v] == 3:
                nu, nv = u, v
                nd = directions[counter]
                break
            counter += 1

        if nd.value == Directions.UP.value:
            action = self.action_id(CellOrientation.HORIZONTAL, nu, nv)
        
        elif nd.value == Directions.LEFT.value:
            action = self.action_id(CellOrientation.VERTICAL, nu, nv)
        
        elif nd.value == Directions.DOWN.value:
            action = self.action_id(CellOrientation.HORIZONTAL, nu+1, nv)
        else : 
            action = self.action_id(CellOrientation.VERTICAL, nu, nv+1)

        return action

    def incount(self, k, i, j, count, loop, cellshist, dirhist):
        count+=1    

        if (k!=1 and self.v_[i][j]<1) :
            dirhist.append(Directions.LEFT)
            if (j>0):  
                if (self.cells[i][j-1]>2): 
                    count+=1
                    loop=True
                    cellshist.append((i,j-1))

                elif (self.cells[i][j-1]>1):
                    cellshist.append((i,j-1))
                    count, loop = self.incount(3,i,j-1, count, loop, cellshist, dirhist)
                
        elif (k!=2 and self.h_[i][j]<1) :
            dirhist.append(Directions.UP)
            if (i>0):
                if (self.cells[i-1][j]>2):
                    count+=1
                    loop=True
                    cellshist.append((i-1,j))
                    
                elif (self.cells[i-1][j]>1):
                    cellshist.append((i-1,j))
                    count, loop = self.incount(4,i-1,j, count, loop, cellshist, dirhist)
            
        elif (k!=3 and self.v_[i][j+1]<1):
            dirhist.append(Directions.RIGHT)
            if (j < self.num_cols-1):

                if (self.cells[i][j+1]>2):
                    count+=1
                    loop=True
                    cellshist.append((i,j+1))

                elif (self.cells[i][j+1]>1):
                    cellshist.append((i,j+1))
                    count, loop = self.incount(1,i,j+1, count, loop, cellshist, dirhist)
                

        elif (k!=4 and self.h_[i+1][j]<1):
            dirhist.append(Directions.DOWN)
            if (i< self.num_rows-1):

                if (self.cells[i+1][j] > 2):
                    count+=1
                    loop= True
                    cellshist.append((i+1,j))

                elif (self.cells[i+1][j] > 1):
                    cellshist.append((i+1,j))
                    count, loop = self.incount(2,i+1,j, count, loop, cellshist, dirhist)

        return count, loop

    def get_available_action(self, state, 
                             cache_chains:Transposition_Table_Chains,  
                             maximizing_player_id):
        
        t1 = t.time()
        # self.obtain_chains(cache_chains, state)
        self.find_all_chains()
        # 1) singletons and doubletons
        possible_actions = self.safe3()
        if possible_actions :
            return possible_actions
        
        # 2) chains
        possible_actions = self.unsafe3()
        if possible_actions :
            return possible_actions

        # 3) random safe edge 
        possible_actions = self.side1()
        if possible_actions : 
            return possible_actions
                
        # 4) last resort
        # ipv alle legale acties terug te geven, 
        # filter degene die de long chain rule niet respcteren
        # dit zorgt er dan weer voor dat de branching factor omlaag gaat 
        # Dit zorgt er dan ook weer voor dat we het niet moeten doen in de eval functie.
        # Dit betekent dat er minder leaf nodes zijn waarvoor we die eval functie moeten berekenen. 
        # het bespaart rekenwerk, en reduces the branching factor. win-win ig. 
        
        # possible_actions = state.legal_actions()
        
        # print("state.legal_actions()")
        # print(possible_actions)

        # long_chain_rule_actions = self.get_lcr_actions(possible_actions, cache_chains, maximizing_player_id)
        
        # print("long_chain_rule_actions")
        # print(long_chain_rule_actions)

        # TODO:
        # soms moet je dus boxes weggeven. 
        # welke actie resulteert in minst aantal boxes te geven aan de opponent. 
        # die lijst van acties dan hier ipv state.legal_actions ?
        # en daaruit dan de lcr rule toepassen ?
        # of da als extra rule naast de lcr rule ? de acties ordenen volgens die nieuwe rule!!!

        # FIXME:,
        # our move ordering only considers non-capturing moves. Of
        # those, the heuristic considers moves that fill in the third edge
        # of a box last, as they leave a capturable box for the opponent. (ibr: dus einde van de lijst volgens mij)
        # 
        # The remaining moves are explored by considering edges in
        # an order radiating outwards from the center of the board. 
        
        # (FIXME: POURQUEE??, of ja i guess so dan, omdat center mss belangerijker is dan rand...)
        
        # This is a very effective heuristic, despite its extreme sim-
        # plicity. On the 4 × 4 solution, for example, this approach
        # reduced the runtime by a factor of 17 over a simple left-to-
        # right, top-to-bottom move order.

        # ordered_actions  = self.moveOrdening(state, long_chain_rule_actions, maximizing_player_id)

        # if ordered_actions:
        #     return ordered_actions
    
        # FIXME: wat als dat niet lukt?
        # actie vinden die de lcr voldoet, dan gwn state.lagal_actions()
        t2 = t.time()
        # print('time')
        # print((t2-t1) * 1000)

        actions = state.legal_actions()
        # if self.num_rows * self.num_cols >= 256 :
            
        #     num_to_pick = min(2, len(actions))
        #     actions = r.sample(actions, num_to_pick)
        #     return actions
        
        # elif self.num_rows * self.num_cols >= 64 : 
            
        #     num_to_pick = min(5, len(actions))
        #     actions = r.sample(actions, num_to_pick)
        #     return actions
        
        # else: 
        num_to_pick = min(10, len(actions))
        actions = r.sample(actions, num_to_pick)
        return actions

    def obtain_chains(self, cache_chains, state): 
        # 0) cache voor de chains. 
        chains_object = cache_chains.get(state.dbn_string())
        if chains_object:
            # print("chains_object obtained from cache.")
            self.chains = chains_object
        else : 
            # print("chains_object calculated.")
            self.find_all_chains()
            cache_chains.set(state.dbn_string(), self.chains)

    # def get_lcr_actions(self, actions, cache_chains, maximizing_player_id):
    #     """
    #     [1] If either m or n is odd, then 
    #         1)  the first player should play to make the number of long chains even.
    #             - als wij de eerste speler zijn, dan filter die actie niet als dit voldaan is.
    #             - als wij de eerste speler zin, dan filter die actie wel als dit voldaan is. 
    #                 (kleiner waarde is beter voor min player, voor opponent)

    #         2) the second player should play to make the number of long chains oneven. 
    #             - als wij de tweede speler zijn, dan extra reward als dit voldaan is.
    #             - als wij de tweede speler zin, dan bestraffen als dit niet voldaan is. 

    #     [2] If both m and n are even, then 
    #         1) The first player should play to make the number of long chains odd. 
    #             - als wij de eerste speler zijn, dan extra reward als dit voldaan is.
    #             - als wij de eerste speler zin, dan bestraffen als dit niet voldaan is.  

    #         2) The second player should play to make the number of long chains even. 
    #             - als wij de eerste speler zijn, dan extra reward als dit voldaan is.
    #             - als wij de eerste speler zin, dan bestraffen als dit niet voldaan is.
    #     """
        
    #     lcr_actions = []
    #     for action in actions: 
    #         cloned_advisor = self.clone()
    #         cloned_advisor.update_action(action, state.current_player())
            
    #         state_string = vectors_to_dbn(self.num_rows, self.num_cols, cloned_advisor.h_, cloned_advisor.v_)
            
    #         chains_object = cache_chains.get(state_string)
    #         if chains_object:
    #             print("chains_object found in cache for child state")
    #             cloned_advisor.chains = chains_object
    #         else : 
    #             print("chains_object calculted for child state")
    #             cloned_advisor.find_all_chains()
    #             cache_chains.set(state_string, self.chains)

    #         total_amount_chains = cloned_advisor.chains["count"]
            
    #         if self.num_rows % 2 != 0 or self.num_cols % 2 != 0:
    #             if total_amount_chains % 2 == 0 and maximizing_player_id == 0: # als de max speler de eerste speler was.
    #                 lcr_actions.append(action)

    #             elif total_amount_chains % 2 != 0 and maximizing_player_id == 1: 
    #                 lcr_actions.append(action)

    #         elif self.num_rows % 2 == 0 and self.num_cols % 2 == 0:        
    #             if total_amount_chains % 2 != 0 and maximizing_player_id == 0: 
    #                 lcr_actions.append(action)

    #             elif total_amount_chains % 2 == 0 and maximizing_player_id == 1:
    #                 lcr_actions.append(action)

    #     return lcr_actions

    # def moveOrdening(self, state, lcr_actions, maximizing_player_id):
    #     # TODO: niet wat er is gebruikt in de paper, zie paper. 
    #     # ordered_lcr_actions = sorted(lcr_actions, 
    #     #                             key=lambda action: 
    #     #                             self.maximal_difference_heuristic(action, state, maximizing_player_id), 
    #     #                             reverse=True) # descending order want beste actie eerst.
    #     # hoe ordenen voor min player!!? 


    #     # our move ordering only considers non-capturing moves. Of
    #     # those, the heuristic considers moves that fill in the third edge
    #     # of a box last, as they leave a capturable box for the opponent. (ibr: dus einde van de lijst volgens mij)
    #     # 
    #     # The remaining moves are explored by considering edges in
    #     # an order radiating outwards from the center of the board. 
                
    #     return ordered_lcr_actions

    # def maximal_difference_heuristic(self, action, state, maximizing_player_id):
    #     cloned_state = state.clone()
    #     cloned_state.apply_action(action)

    #     #TODO: ik kan hier ook gwn een eigen cache weer voor hebben. 

    #     state_matrix = game_state_tensor(cloned_state)
    #     points_max_player = sum(state_matrix[maximizing_player_id + 1, :, 2])
    #     points_min_player = sum(state_matrix[2 - maximizing_player_id, :, 2])

    #     return points_max_player - points_min_player
    
    # def sides1(self):
    #     safe_actions = []

    #     for i in range(self.num_rows + 1):
    #         for j in range(self.cols):
    #             if self.safe_hedge(i, j):
    #                 action = SA.action_id(CellOrientation.HORIZONTAL, i ,j)
    #                 safe_actions.append(action)
        
    #     for i in range(self.num_rows):
    #         for j in range(self.cols + 1):
    #             if self.safe_vedge(i, j):
    #                 action = SA.action_id(CellOrientation.VERTICAL, i ,j)
    #                 safe_actions.append(action)
                
    #     np.random.shuffle(safe_actions)
    #     return safe_actions


    def side1(self, seed=None):
        if seed is not None:
            r.seed(seed)

        if r.random() < 0.5: 
            orien = CellOrientation.HORIZONTAL
        else :
            orien = CellOrientation.VERTICAL

        i = r.randint(0, self.num_rows - 1)
        j = r.randint(0, self.num_cols - 1)

        if orien.value == CellOrientation.HORIZONTAL.value:
            pos = self.rand_hedge(i, j)
            if pos:
                x, y = pos
                action = self.action_id(CellOrientation.HORIZONTAL, x, y)
                return [action]
            
            else:
                pos = self.rand_vedge(i, j)
                if pos:
                    x, y = pos
                    action = self.action_id(CellOrientation.VERTICAL, x, y)
                    return [action]
                
        else:
            pos = self.rand_vedge(i, j)
            if pos:
                x, y = pos
                action = self.action_id(CellOrientation.VERTICAL, x, y)
                return [action]
            else:
                pos = self.rand_hedge(i, j)
                if pos:
                    x, y = pos
                    action = self.action_id(CellOrientation.HORIZONTAL, x, y)
                    return [action]
                
        return None

    def rand_hedge(self, i, j):
        x = i
        y = j
        while True:
            if self.safe_hedge(x, y):
                return x, y
            else:
                y += 1
                if y == self.num_cols:
                    y = 0
                    x += 1
                    if x > self.num_rows:
                        x = 0
            if x == i and y == j:
                break
        return None

    def rand_vedge(self, i, j):
        x = i
        y = j
        while True:
            if self.safe_vedge(x, y):
                return x, y 
            else:
                y += 1
                if y > self.num_cols:
                    y = 0
                    x += 1
                    if x == self.num_rows:
                        x = 0
            if x == i and y == j:
                break
        return None

    def safe_hedge(self, i, j):
        if self.h_[i][j] < 1:
            if i == 0:
                if self.cells[i][j] < 2:
                    return True
            elif i == self.num_rows:
                if self.cells[i - 1][j] < 2:
                    return True
            elif self.cells[i][j] < 2 and self.cells[i - 1][j] < 2:
                return True
        return False

    def safe_vedge(self, i, j):
        if self.v_[i][j] < 1:
            if j == 0:
                if self.cells[i][j] < 2:
                    return True
            elif j == self.num_cols:
                if self.cells[i][j - 1] < 2:
                    return True
            elif self.cells[i][j] < 2 and self.cells[i][j - 1] < 2:
                return True
        return False

if __name__ == "__main__":
    SA = StrategyAdvisor(10,10)
    # example_paper(SA)
    # horizontal_closed_chain(SA)
    # vertical_closed_chain(SA)
    # SA.unsafe3()
    # closed_chain_data = SA.chains["closed"][0]
    # closed_chain = closed_chain_data["chain"]
    # closed_dir   = closed_chain_data["directions"]
    # SA.get_double_half_haerted(closed_chain,closed_dir)
