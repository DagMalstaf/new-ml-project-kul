import torch
import torch_geometric.data as geom_data

def state_to_graph_data(state):
    game = state.get_game()
    cols, rows = game.get_parameters()["num_cols"], game.get_parameters()["num_rows"]
    num_nodes = cols*rows

    x = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for i in range(rows):
        for j in range(cols):
            node_index = i * cols + j
            owner = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i, col=j, part='c')
            if state.current_player() == 1 & owner != 0:
                owner = 3 - owner
            x[node_index, 0] = owner
            x[node_index, 1] = 4 - getFilledLines(game,state.observation_tensor(),i,j)

    edge_index = []
    edge_attr = []

    for i in range(rows):
        for j in range(cols):
            node_index = i * cols + j
            if i == 0:
                edge_index.append([node_index, node_index])
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i, col=j, part='h')
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                edge_attr.append(edge_state)
            
            if i < rows - 1:
                bottom_neighbor_index = (i+1)*(cols) + j
                edge_index.append([node_index, bottom_neighbor_index])
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i+1, col=j, part='h')
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                edge_attr.append(edge_state)

            if i == rows - 1:
                edge_index.append([node_index, node_index])
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i+1, col=j, part='h')
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                edge_attr.append(edge_state)

    for i in range(rows):
        for j in range(cols):
            node_index = i * cols + j

            if j == 0:
                edge_index.append([node_index, node_index])
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i, col=j, part='v')
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                edge_attr.append(edge_state)

            if j < cols - 1:
                right_neighbor_index = i*(cols) + (j+1)
                edge_index.append([node_index, right_neighbor_index])
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i, col=j+1, part='v')
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                edge_attr.append(edge_state)
            
            if j ==  cols - 1:                
                edge_index.append([node_index, node_index])
                edge_state = get_observation_state(game=game,obs_tensor=state.observation_tensor(), row=i, col=j+1, part='v')
                
                if state.current_player() == 1 & edge_state != 0:
                    edge_state = 3 - edge_state
                edge_attr.append(edge_state)
            
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.int64)

    batch = torch.zeros(num_nodes, dtype=torch.int64)

    return geom_data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr,batch=batch)


def part2num(part):
    p = {'h': 0, 'horizontal': 0,  
         'v': 1, 'vertical':   1,  
         'c': 2, 'cell':       2}  
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
    num_parts = 3   
    num_states = 3 
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