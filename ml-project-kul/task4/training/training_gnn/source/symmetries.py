import numpy as np
import util as s


# There are a number of trivial symmetries in Dots-And-Boxes
# that reduce the problem space. The mirror image of a state is
# also a legal game whose optimal strategy mirrors that of the
# current state. All Dots-And-Boxes instances have horizon-
# tal and vertical symmetry, and square boards have diagonal
# symmetry.


# only needed for some tests:
def _find_all_symmetries(state, num_rows, num_cols):
    """shouldnt be used, `if elif` etc. is better. """
    s1 = check_horizontal(state, num_rows, num_cols)
    s2 = check_vertical(state, num_rows, num_cols)
    s3 = check_hv(state, num_rows, num_cols)
    s4 = check_vh(state, num_rows, num_cols)
    if num_rows == num_cols :
        s5 = check_diag1(state, num_rows, num_cols)
        s6 = check_diag2(state, num_rows, num_cols)
        s7 = check_diag2(state, num_rows, num_cols)
        return [s1, s2, s3, s4, s5 , s6, s7]
    else :
        return [s1, s2, s3, s4]

def check_horizontal(state, num_rows, num_cols):
    dbn = state.dbn_string()
    h_matrix, v_matrix = s.dbn_to_vectors(num_rows, num_cols, dbn)

    new_h_list = np.flipud(h_matrix)
    new_v_list = np.flipud(v_matrix)

    return s.vectors_to_dbn(num_rows, num_cols, new_h_list, new_v_list)

def check_vertical(state, num_rows, num_cols):
    dbn = state.dbn_string()
    h_matrix, v_matrix = s.dbn_to_vectors(num_rows, num_cols, dbn)

    new_h_list = np.fliplr(h_matrix)
    new_v_list = np.fliplr(v_matrix)

    return s.vectors_to_dbn(num_rows, num_cols,new_h_list, new_v_list)

def check_hv(state, num_rows, num_cols):
    dbn = state.dbn_string()
    h_matrix, v_matrix = s.dbn_to_vectors(num_rows, num_cols, dbn)

    h_matrix = np.flipud(h_matrix)
    v_matrix = np.flipud(v_matrix)

    new_h_list = np.fliplr(h_matrix)
    new_v_list = np.fliplr(v_matrix)

    return s.vectors_to_dbn(num_rows, num_cols,new_h_list, new_v_list)

def check_vh(state, num_rows, num_cols):
    dbn = state.dbn_string()
    h_matrix, v_matrix = s.dbn_to_vectors(num_rows, num_cols, dbn)

    new_h_list = np.fliplr(h_matrix)
    new_v_list = np.fliplr(v_matrix)

    h_matrix = np.flipud(new_h_list)
    v_matrix = np.flipud(new_v_list)

    return s.vectors_to_dbn(num_rows, num_cols, h_matrix, v_matrix)

# only for square boards
def check_diag1(state, num_rows, num_cols):
    """ rotate matrix counterclockwise 270 degrees"""
    dbn = state.dbn_string()
    h_matrix, v_matrix = s.dbn_to_vectors(num_rows, num_cols, dbn)
    
    new_h_list = np.rot90(v_matrix, 3)
    new_v_list = np.rot90(h_matrix, 3)
    
    return s.vectors_to_dbn(num_rows, num_cols, new_h_list, new_v_list)

def check_diag2(state, num_rows, num_cols):
    """ rotate matrix counterclockwise 90 degrees"""
    dbn = state.dbn_string()
    h_matrix, v_matrix = s.dbn_to_vectors(num_rows, num_cols, dbn)

    new_h_list = np.rot90(v_matrix, 1)
    new_v_list = np.rot90(h_matrix, 1)
    
    return s.vectors_to_dbn(num_rows, num_cols, new_h_list, new_v_list)


# only for square boards
def check_h_diag1(state, num_rows, num_cols):
    dbn = state.dbn_string()
    h_matrix, v_matrix = s.dbn_to_vectors(num_rows, num_cols, dbn)

    new_h_list = np.flipud(h_matrix)
    new_v_list = np.flipud(v_matrix)

    h_list = np.rot90(new_v_list, 3)
    v_list = np.rot90(new_h_list, 3)
    
    return s.vectors_to_dbn(num_rows, num_cols, h_list, v_list)