from enum import Enum
import numpy as np

class CellOrientation(Enum):  
    HORIZONTAL = 1
    VERTICAL = 2

class Directions(Enum):  
    UP = 1
    DOWN = 2
    RIGHT = 3
    LEFT = 4

# TE TRAAG
# def vectors_to_dbn(num_rows, num_cols, h_, v_):
#     """
#     Constructs a string as follows : 
#     [b | for r in [0,num_rows+1], for c in [0,num_cols]:
#          b=1 if horizontal line[r,c] set else 0] 
#     ++
#     [b | for r in [0,num_rows_], for c in [0,num_cols+1]:
#          b=1 if vertical line[r,c] set else 0]
#     """
#     dbn = ""

#     for r in range(num_rows + 1):
#         for c in range(num_cols):
#             dbn += "1" if h_[r][c] else "0"

#     for r in range(num_rows):
#         for c in range(num_cols + 1):
#             dbn += "1" if v_[r][c] else "0"

#     return dbn


def vectors_to_dbn(num_rows, num_cols, h_, v_):
    """
    Constructs a string as follows : 
    [b | for r in [0,num_rows+1], for c in [0,num_cols]:
         b=1 if horizontal line[r,c] set else 0] 
    
    ++
    
    [b | for r in [0,num_rows_], for c in [0,num_cols+1]:
         b=1 if vertical line[r,c] set else 0]
    """
    matrix1 = np.array(h_)
    matrix2 = np.array(v_)

    flattened_matrix1 = matrix1.flatten()
    flattened_matrix2 = matrix2.flatten()

    h_string = ''.join(map(str, flattened_matrix1))
    v_string = ''.join(map(str, flattened_matrix2))

    return h_string + v_string


def pvector_for_tensor(tensor):
    p = np.zeros(len(tensor[0]))

    # Player 1
    for i in range(len(tensor[1])):
        if tensor[1][i][2] == 1:
            p[i] = 1

    # Player 2
    for i in range(len(tensor[2])):
        if tensor[2][i][2] == 1:
            p[i] = 2
 
    return p

# def hvector_for_tensor(tensor, num_dots):
#     h = np.zeros(num_dots)

#     # Player 1
#     for i in range(num_dots):
#         if tensor[1][i][0] == 1:
#             h[i] = 1

#     # Player 2
#     for i in range(num_dots):
#         if tensor[2][i][0] == 1:
#             h[i] = 2
 
#     return h

# def vectors_to_dbn(num_rows, num_cols, h_, v_):
#     num_dots = (nrows + 1) * (ncols + 1) 
#     h = np.zeros(num_dots, dtype=int)
#     v = np.zeros(num_dots, dtype=int)

#     # Player 1
#     player1_indices = np.where(tensor[1][:, 0] == 1)[0]
#     h[player1_indices] = 1

#     player1_indices = np.where(tensor[1][:, 1] == 1)[0]
#     v[player1_indices] = 1

#     # Player 2
#     player2_indices = np.where(tensor[2][:, 0] == 1)[0]
#     h[player2_indices] = 2

#     player2_indices = np.where(tensor[2][:, 1] == 1)[0]
#     v[player2_indices] = 2

#     h = h.reshape(nrows+1, ncols+1)[:, :-1].ravel()
#     v = v.reshape(nrows+1, ncols+1)[:-1, :].ravel()

#     return "".join(map(str, h)) + "".join(map(str, v))

# def h_and_v_vectors_from_tensor(tensor, nrows, ncols):
#     num_dots = (nrows + 1) * (ncols + 1) 
#     hv = np.zeros(2 * num_dots, dtype=int)

#     # Player 1
#     player1_indices = np.where(tensor[1][:, 0:1] == 1)[0]
#     hv[player1_indices] = 1
#     # Player 2
#     player2_indices = np.where(tensor[2][:, 0:1] == 1)[0]
#     hv[player2_indices] = 2

#     # h = hv[:num_dots].reshape(nrows+1, ncols+1)[:, :-1].ravel()
#     # v = hv[num_dots:].reshape(nrows+1, ncols+1)[:-1, :].ravel()

#     h = hv[:num_dots]
#     h = np.delete(h, np.arange(ncols, num_dots, ncols+1))

#     v = hv[num_dots:]
#     v = np.delete(v, np.arange(num_dots - ncols, num_dots))

#     return "".join(map(str, h)) + "".join(map(str, v))


def box_state_for_pvector(pvector, nrows, ncols):
    pmatrix = pvector.reshape(nrows+1, ncols+1)
    result = pmatrix[:-1, :-1].ravel()
    return result

def dbn_to_vectors(num_rows, num_cols, dbn:str):
    """
    Create horizontal and vertical edge vectors from the Dots-and-Boxes Notation.
    """
    
    h_ = [[0] * num_cols for _ in range(num_rows + 1)]
    v_ = [[0] * (num_cols + 1) for _ in range(num_rows)]
    num_moves = 0
    
    idx = 0
    for row in range(num_rows + 1):
        for col in range(num_cols):
            if dbn[idx] == '1':
                h_[row][col] = 1
                num_moves += 1
            idx += 1
    
    for row in range(num_rows):
        for col in range(num_cols + 1):
            if dbn[idx] == '1':
                v_[row][col] = 1
                num_moves += 1
            idx += 1
    
    max_moves = (num_rows + 1) * num_cols + num_rows * (num_cols + 1)
    assert num_moves <= max_moves, "Number of moves exceeds maximum possible moves"

    return h_, v_