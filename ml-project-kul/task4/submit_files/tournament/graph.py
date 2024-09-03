import torch
import torch_geometric.data as geom_data
import numpy as np
import pyspiel


def state_to_graph_data(state):
    """
    This function takes a state and returns a PyTorch Geometric Data object
    in the perspective of the current player of the state (so the first player)
    """
    game = state.get_game()
    cols, rows = game.get_parameters()["num_cols"], game.get_parameters()["num_rows"]
    num_nodes = cols*rows
    
    # Node features
    x = torch.zeros((num_nodes, 5), dtype=torch.float32) 
    for i in range(rows):
        for j in range(cols):
            node_index = i * cols + j
            owner = get_observation_state(game=game, obs_tensor=state.observation_tensor(), row=i, col=j, part='c')
            
            # Ownership perspective
            if state.current_player() == 1 and owner != 0:
                owner = 3 - owner
            x[node_index, 0] = owner
            
            # Number of unfilled sides - also the amount of strings that are connected to this node, filled line means that string is not connected
            x[node_index, 1] = 4 - getFilledLines(game, state.observation_tensor(), i, j)
            
            # New Features for Chain Strategy
            x[node_index, 2] = is_part_of_chain(i, j, state)  # Binary chain membership
            x[node_index, 3] = chain_length(i, j, state)  # Chain length if part of a chain
            x[node_index, 4] = potential_chain(i, j, state)  # Potential to become part of a chain


    # Edge indices
    # Coins and string representation
    # for edges that have strings to no other node, we add a edge to the node itself
    # try to follow the same order as the actions in openspiel, first all the horizontal edges, then all the vertical edges
    edge_index = []
    edge_attr = []
    # edge attributes -> state of the graph

    # Top and bottom strings first (vertical strings)
    for i in range(rows):
        for j in range(cols):
            node_index = i * cols + j
            if i == 0:
                edge_index.append([node_index, node_index])
                edge_state = get_observation_state(game=game, obs_tensor=state.observation_tensor(), row=i, col=j, part='h')
                if state.current_player() == 1 and edge_state != 0:
                    edge_state = 3 - edge_state
                
                chain_potential = chain_forming_potential(i, j, 'h', state)
                critical_edge = critical_edge_indicator(i, j, 'h', state)
                
                edge_attr.append([edge_state, chain_potential, critical_edge])
            
            if i < rows - 1:
                bottom_neighbor_index = (i + 1) * cols + j
                edge_index.append([node_index, bottom_neighbor_index])
                edge_state = get_observation_state(game=game, obs_tensor=state.observation_tensor(), row=i + 1, col=j, part='h')
                if state.current_player() == 1 and edge_state != 0:
                    edge_state = 3 - edge_state
                
                chain_potential = chain_forming_potential(i, j, 'h', state)
                critical_edge = critical_edge_indicator(i, j, 'h', state)
                
                edge_attr.append([edge_state, chain_potential, critical_edge])


            if i == rows - 1:
                edge_index.append([node_index, node_index])
                edge_state = get_observation_state(game=game, obs_tensor=state.observation_tensor(), row=i + 1, col=j, part='h')
                if state.current_player() == 1 and edge_state != 0:
                    edge_state = 3 - edge_state
                
                chain_potential = chain_forming_potential(i, j, 'h', state)
                critical_edge = critical_edge_indicator(i, j, 'h', state)
                
                edge_attr.append([edge_state, chain_potential, critical_edge])


    # Left and right strings (horizontal strings)
    for i in range(rows):
        for j in range(cols):
            node_index = i * cols + j

            if j == 0:
                edge_index.append([node_index, node_index])
                edge_state = get_observation_state(game=game, obs_tensor=state.observation_tensor(), row=i, col=j, part='v')
                if state.current_player() == 1 and edge_state != 0:
                    edge_state = 3 - edge_state
                
                chain_potential = chain_forming_potential(i, j, 'v', state)
                critical_edge = critical_edge_indicator(i, j, 'v', state)
                
                edge_attr.append([edge_state, chain_potential, critical_edge])

            if j < cols - 1:
                # Horizontal edge
                right_neighbor_index = i * cols + (j + 1)
                edge_index.append([node_index, right_neighbor_index])
                # set the edge attribute if the edge is filled
                edge_state = get_observation_state(game=game, obs_tensor=state.observation_tensor(), row=i, col=j + 1, part='v')
                if state.current_player() == 1 and edge_state != 0:
                    edge_state = 3 - edge_state
                
                chain_potential = chain_forming_potential(i, j, 'v', state)
                critical_edge = critical_edge_indicator(i, j, 'v', state)
                
                edge_attr.append([edge_state, chain_potential, critical_edge])
            
            if j == cols - 1:                
                edge_index.append([node_index, node_index])
                # set the edge attribute if the edge is filled
                edge_state = get_observation_state(game=game, obs_tensor=state.observation_tensor(), row=i, col=j + 1, part='v')
                if state.current_player() == 1 and edge_state != 0:
                    edge_state = 3 - edge_state
                
                chain_potential = chain_forming_potential(i, j, 'v', state)
                critical_edge = critical_edge_indicator(i, j, 'v', state)
                
                edge_attr.append([edge_state, chain_potential, critical_edge])

            
    # turn edge index into a tensor
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()
    # turn edge attribute into a tensor
    edge_attr = torch.tensor(edge_attr, dtype=torch.int64)


    # Batch information
    batch = torch.zeros(num_nodes, dtype=torch.int64)

    return geom_data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr,batch=batch) # batch??, add dummy node?


def part2num(part):
    p = {'h': 0, 'horizontal': 0,  # Who has set the horizontal line (top of cell)
         'v': 1, 'vertical':   1,  # Who has set the vertical line (left of cell)
         'c': 2, 'cell':       2}  # Who has won the cell
    return p.get(part, part)
def state2num(state):
    s = {'e':  0, 'empty':   0,
         'p1': 1, 'player1': 1,
         'p2': 2, 'player2': 2}
    return s.get(state, state)
def num2state(state):
    s = {0: 'empty', 1: 'player1', 2: 'player2'}
    return s.get(state, state)


def get_observation(game,obs_tensor, state, row, col, part): 
    num_rows, num_cols = game.get_parameters()["num_rows"], game.get_parameters()["num_cols"]
    num_cells = (num_rows + 1) * (num_cols + 1)
    num_parts = 3   # (horizontal, vertical, cell)
    num_states = 3  # (empty, player1, player2)
    state = state2num(state)
    part = part2num(part)
    idx =   part \
          + (row * (num_cols + 1) + col) * num_parts  \
          + state * (num_parts * num_cells)
    return obs_tensor[idx]

def get_observation_state(game,obs_tensor, row, col, part):
    is_state = None
    for state in range(3):
        if get_observation(game, obs_tensor, state, row, col, part) == 1.0:
            is_state = state
    return is_state

def getFilledLines(game,obs_tensor,row,col):
    connections = 4
    connections -= get_observation(game,obs_tensor,state2num('empty'),row,col,'h')
    connections -= get_observation(game,obs_tensor,state2num('empty'),row,col,'v')
    connections -= get_observation(game,obs_tensor,state2num('empty'),row,col+1,'v')
    connections -= get_observation(game,obs_tensor,state2num('empty'),row+1,col,'h')
    return connections

def edges_to_actions(state,edges):
    game = state.get_game()
    num_rows = game.get_parameters()["num_rows"]
    actions = []
    first_row = []
    second_row = []
    for i in range(num_rows*2):
        if i % 2 == 0:
            first_row.append(edges[i])
        else:
            second_row.append(edges[i])
    actions += first_row
    actions += second_row
    actions += edges[num_rows*2:]
    return actions

def is_part_of_chain(row, col, state):
    game = state.get_game()
    num_unfilled_edges = getFilledLines(game, state.observation_tensor(), row, col)
    
    # If the box has exactly 1 or 2 unfilled edges, it's part of a chain
    return 1 if num_unfilled_edges in [1, 2] else 0

def chain_length(row, col, state):
    def traverse_chain(row, col, visited):
        if (row, col) in visited or is_part_of_chain(row, col, state) == 0:
            return 0
        
        visited.add((row, col))
        length = 1  # Start with the current box
        game = state.get_game()
        
        # Check all four directions: top, bottom, left, right
        neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        
        for r, c in neighbors:
            if 0 <= r < game.get_parameters()["num_rows"] and 0 <= c < game.get_parameters()["num_cols"]:
                length += traverse_chain(r, c, visited)
        
        return length

    visited = set()
    return traverse_chain(row, col, visited)

def potential_chain(row, col, state):

    game = state.get_game()
    num_rows, num_cols = game.get_parameters()["num_rows"], game.get_parameters()["num_cols"]

    # Check if the current box is already part of a chain
    if is_part_of_chain(row, col, state) == 1:
        return 1

    # Check if this box is on the verge of becoming part of a chain
    num_unfilled_edges = getFilledLines(game, state.observation_tensor(), row, col)
    if num_unfilled_edges == 2:
        # A box with exactly two unfilled edges has high potential to join a chain
        return 1

    # Evaluate neighbors to see if they can contribute to this box forming a chain
    neighbors = [
        (row - 1, col),  # Above
        (row + 1, col),  # Below
        (row, col - 1),  # Left
        (row, col + 1)   # Right
    ]

    for r, c in neighbors:
        if 0 <= r < num_rows and 0 <= c < num_cols:
            # If a neighboring box is part of a chain, and the current box can connect to it, this box has potential
            if is_part_of_chain(r, c, state) == 1 and getFilledLines(game, state.observation_tensor(), r, c) == 2:
                return 1

    return 0


def chain_forming_potential(row, col, part, state):
    game = state.get_game()
    num_rows, num_cols = game.get_parameters()["num_rows"], game.get_parameters()["num_cols"]

    # Check if the edge is currently unfilled
    if get_observation_state(game, state.observation_tensor(), row, col, part) != 0:
        return 0
    
    if part == 'h':  # Horizontal edge
        neighbors = [(row, col), (row - 1, col)] if row > 0 else [(row, col)]
    else:  # Vertical edge
        neighbors = [(row, col), (row, col - 1)] if col > 0 else [(row, col)]

    # Check if any of the neighboring boxes are part of a chain or could become part of one
    for r, c in neighbors:
        if 0 <= r < num_rows and 0 <= c < num_cols:
            if is_part_of_chain(r, c, state) == 1 or potential_chain(r, c, state) == 1:
                return 1
    return 0


def critical_edge_indicator(row, col, part, state):
    game = state.get_game()
    num_rows, num_cols = game.get_parameters()["num_rows"], game.get_parameters()["num_cols"]

    # Check if the edge is currently unfilled
    if get_observation_state(game, state.observation_tensor(), row, col, part) != 0:
        return 0
    
    if part == 'h':  # Horizontal edge
        neighbors = [(row, col), (row - 1, col)] if row > 0 else [(row, col)]
    else:  # Vertical edge
        neighbors = [(row, col), (row, col - 1)] if col > 0 else [(row, col)]

    # Check if filling this edge would complete a box that is part of a chain
    for r, c in neighbors:
        if 0 <= r < num_rows and 0 <= c < num_cols:
            if getFilledLines(game, state.observation_tensor(), r, c) == 3:
                if is_part_of_chain(r, c, state) == 1 or potential_chain(r, c, state) == 1:
                    return 1

    return 0



