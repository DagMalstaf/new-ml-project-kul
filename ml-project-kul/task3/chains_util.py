def chain_is_open(chain, strat_box_dict):
    count_of_threes = int()
    count_of_ones = int()
    count_of_zeroes = int()

    for vertex in chain:
        if len(strat_box_dict[vertex]) == 1:
            count_of_threes += 1
        if len(strat_box_dict[vertex]) == 3:
            count_of_ones += 1
        if len(strat_box_dict[vertex]) == 4:
            count_of_zeroes += 1
    if (count_of_threes - 1) >= (2*count_of_zeroes + count_of_ones):
        return True
    else:
        return False


def take_chain(chain, strat_box_dict):
    for vertex in chain:
        if len(strat_box_dict[vertex]) == 1:
            possible_edge = list(strat_box_dict[vertex])
            requested_edge = possible_edge[0]
            return (requested_edge, vertex)
    return (None, None)

def open_chain(chain, strat_box_dict):
    for vertex in chain:
        if len(strat_box_dict[vertex]) == 2:
            possible_edges = list(strat_box_dict[vertex])
            requested_edge = possible_edges[0]
            return requested_edge
    return None

def get_random_edge(game_graph, box_dict):
    from random import randint
    possible_edges = set()
    for vertex in box_dict:
        if len(box_dict[vertex]) > 0:
            possible_edges.update(box_dict[vertex])
    edge_index = randint(0, len(possible_edges)-1)
    possible_edges = list(possible_edges)
    chosen_edge = possible_edges[edge_index]
    return chosen_edge 

def coords_to_vertex(game_dict, coordinates):
    for vertex_identifier, vertex_coordinates in game_dict.items():
        if coordinates == vertex_coordinates:
            return vertex_identifier
    return -1

def vertex_to_coords(game_dict, vertex):
    for vertex_identifier, vertex_coordinates in game_dict.items():
        if vertex == vertex_identifier:
            return vertex_coordinates
    return -1

def draw_line(serial_in, serial_out, requested_edge):
    global game_graph, game_dict, box_dict
    global strat_graph, edge_intersect_dict
    global num_moves, game_move, computer_move
  
    if game_graph.is_edge(requested_edge) or \
        requested_edge[0] == requested_edge[1]:
        if error: return -1
        return 0 

    num_moves -= 1 
    game_graph.add_edge(requested_edge)
    rev_requested_edge = (requested_edge[1], requested_edge[0])
  
    for edge, intersecting_edge in edge_intersect_dict.items():
        if requested_edge == intersecting_edge:
            strat_graph.remove_edge(edge)
            edge_intersect_dict[edge] = None

        elif rev_requested_edge == intersecting_edge:
            strat_graph.remove_edge(edge)
            edge_intersect_dict[edge] = None

    return 1

def get_boxes(serial_in, serial_out, requested_edge):
    global box_dict
    global strat_box_dict

    boxes = list()
    rev_requested_edge = (requested_edge[1], requested_edge[0])
    for box in box_dict:

        if requested_edge in box_dict[box]:
            box_dict[box].remove(requested_edge)
          
            if len(box_dict[box]) == 0:
                boxes.append(box)

        elif rev_requested_edge in box_dict[box]:
            box_dict[box].remove(rev_requested_edge)

            if len(box_dict[box]) == 0:
                boxes.append(box)

    for box in strat_box_dict:
        if requested_edge in strat_box_dict[box]:
            strat_box_dict[box].remove(requested_edge)

        elif rev_requested_edge in strat_box_dict[box]:
            strat_box_dict[box].remove(rev_requested_edge)

    return (boxes, len(boxes))

def process_line(serial_in, serial_out, requested_edge):
    global game_over, error
    global game_dict
    global game_move, computer_move

    line_drawn = draw_line(serial_in, serial_out, requested_edge)
    if line_drawn < 1:
        return (game_over, error)
    (boxes, num_boxes) = get_boxes(serial_in, serial_out, requested_edge)
    return (game_over, error)

def computer_turn(serial_in, serial_out):
    global game_graph, box_dict
    global strat_graph, stored_chain, edge_intersect_dict, computer_is_first
    global strat_box_dict
    global num_columns, num_rows

    num_dots = ((num_columns + 1) * (num_rows + 1))
    if len(stored_chain) > 0:

        (requested_edge, chosen) = take_chain(stored_chain, strat_box_dict)
        if not chosen is None:
            stored_chain.remove(chosen)
        else:
            stored_chain = list()

        if not requested_edge is None:
            process_line(serial_in, serial_out, requested_edge)
            return requested_edge, True

    components = get_components(strat_graph)
    subgraph_list = list()
    for vertex_set in components:
        subgraph = UndirectedAdjacencyGraph()

        if len(vertex_set) == 1:
            continue

        for v in vertex_set:
            if not subgraph.is_vertex(v):
                subgraph.add_vertex(v)
        subgraph_list.append(subgraph)

        for edge, intersect_edge in edge_intersect_dict.items():
            if not intersect_edge is None:
                if subgraph.is_vertex(edge[0]) and subgraph.is_vertex(edge[1]):
                    subgraph.add_edge(edge)

    long_chains = list()
    short_chains = list()
    for subgraph in subgraph_list:
        if subgraph.is_cyclic():
            continue
        else:
            if len(subgraph.vertices()) >= 3:
                long_chains.append({v for v in subgraph.vertices()})
            else:
                short_chains.append({v for v in subgraph.vertices()})
    if len(long_chains) > 0:
        sorted_long_chains = sorted(long_chains, key=len, reverse=True)

        for chain in sorted_long_chains:
            if chain_is_open(chain, strat_box_dict):
                stored_chain = chain
                (requested_edge, chosen) = \
                    take_chain(stored_chain, strat_box_dict)

                if not chosen is None:
                    stored_chain.remove(chosen)
                else:
                    stored_chain = list()
                if not requested_edge is None:
                    process_line(serial_in, serial_out, requested_edge)
                    return requested_edge, True

    if (num_dots + len(long_chains)) % 2 == 0 and computer_is_first:
        computer_has_control = True
    elif (num_dots + len(long_chains)) % 2 != 0 and not computer_is_first:
        computer_has_control = True
    else:
        computer_has_control = False
      
    if computer_has_control:
        for chain in short_chains:
            if chain_is_open(chain, strat_box_dict):
                stored_chain = chain
                (requested_edge, chosen) = \
                    take_chain(stored_chain, strat_box_dict)

                if not chosen is None:
                    stored_chain.remove(chosen)
                else:
                    stored_chain = list()
                if not requested_edge is None:
                    process_line(serial_in, serial_out, requested_edge)
                    return requested_edge, True
    else:
        for chain in short_chains:
            if not chain_is_open(chain, strat_box_dict):
                requested_edge = open_chain(stored_chain, strat_box_dict)
                if not requested_edge is None:
                    return process_line(serial_in, serial_out, requested_edge), True
    requested_edge = get_random_edge(game_graph, box_dict)
    return requested_edge, False


class UndirectedAdjacencyGraph:
    def __init__(self):
        self._vertices = dict()

    def add_vertex(self, v):
        if v not in self._vertices:
            self._vertices[v] = list()
        else:
            raise RuntimeError("Bad argument:"
                               " Vertex {} already in the graph".format(v))

    def is_vertex(self, v):
        return v in self._vertices

    def add_edge(self, e):
        if not self.is_vertex(e[0]):
            raise RuntimeError("Attempt to create an edge with"
                                  " non-existent vertex: {}".format(e[0]))
        if not self.is_vertex(e[1]):
            raise RuntimeError("Attempt to create an edge with"
                                  "non-existent vertex: {}".format(e[1]))

        if not e[1] in self._vertices[e[0]]:
            self._vertices[e[0]].append(e[1])
        if not e[0] in self._vertices[e[1]]:
            self._vertices[e[1]].append(e[0])

    def is_edge(self, e):

        if (e[1] in self._vertices[e[0]]) or (e[0] in self._vertices[e[1]]):
            return True
        else:
            return False

    def remove_edge(self, e):
        if self.is_edge(e):
            self._vertices[e[0]].remove(e[1])
            self._vertices[e[1]].remove(e[0])

    def neighbours(self, v):
        return self._vertices[v]

    def vertices(self):
        return set(self._vertices.keys())

    def clear(self):
        self._vertices = dict()

    def is_cyclic_util(self, v, visited, parent):
        visited[v] = True 
        for i in self.neighbours(v):
            if visited[i] == False:
                if (self.is_cyclic_util(i, visited, v)):
                    return True
                  
            elif parent != i:
                return True

        return False

    def is_cyclic(self):
        visited = {v:False for v in self.vertices()}

        for i in self.vertices():
            if visited[i] == False:
                if (self.is_cyclic_util(i, visited, -1)) == True:
                    return True

        return False

def breadth_first_search(g, v):
    import queue

    todolist = queue.deque([v])
    reached = {v}
    while todolist:
        u = todolist.popleft()
        for w in g.neighbours(u):
            if w not in reached:
                reached.add(w)  
                todolist.append(w) 

    return frozenset(reached)

def get_components(g):
    component_set = set()
    for v in g.vertices():
        component = breadth_first_search(g, v)

        if component in component_set:
            continue

        else:
            component_set.add(component)

    return component_set

import sys

game_over = bool() 
error = bool() 

game_graph = UndirectedAdjacencyGraph() 
game_dict = dict()
box_dict = dict()

strat_graph = UndirectedAdjacencyGraph()
strat_dict = dict()
strat_box_dict = dict()
stored_chain = list()
computer_is_first = bool()

num_columns = int()
num_rows = int()
num_moves = int() 

game_move = int()
computer_move = int()

def build_game_graph(num_columns, num_rows):
    game_graph = UndirectedAdjacencyGraph()
    game_dict = dict()
    vertex_number = int()
    for i in range(num_rows+1):
        for j in range(num_columns+1):
            game_graph.add_vertex(vertex_number)
            game_dict[vertex_number] = (j, i)
            vertex_number += 1

    box_dict = dict()
    strat_box_dict = dict()
    strat_vertex = int()
    game_vertex = int()
    for i in range(num_rows):
        for j in range(num_columns):
            edge1 = (game_vertex, game_vertex+1)
            edge2 = (game_vertex, game_vertex+num_columns+1)
            edge3 = (game_vertex+1, game_vertex+num_columns+2)
            edge4 = (game_vertex+num_columns+1, game_vertex+num_columns+2)

            box_dict[game_vertex] = {edge1, edge2, edge3, edge4}
            strat_box_dict[strat_vertex] = {edge1, edge2, edge3, edge4}
          
            strat_vertex += 1
            if j == num_columns-1:
                game_vertex += 2
            else:
                game_vertex += 1

    return (game_graph, game_dict, box_dict, strat_box_dict)

def build_strat_graph(game_dict, num_columns, num_rows):
    strat_graph = UndirectedAdjacencyGraph()

    for v in range(num_columns*num_rows):
        strat_graph.add_vertex(v)

    strat_dict = dict()
    vertex_number = int()
    for i in range(num_rows):
        for j in range(num_columns):
            if i == (num_rows - 1) and j == (num_columns - 1):
                pass

            elif i == (num_rows - 1):
                edge = (vertex_number, vertex_number+1)
                strat_graph.add_edge(edge)

                if vertex_number not in strat_dict:
                    strat_dict[vertex_number] = set()
                strat_dict[vertex_number].add(edge)

                if vertex_number+1 not in strat_dict:
                    strat_dict[vertex_number+1] = set()
                strat_dict[vertex_number+1].add(edge)

            elif j == (num_columns - 1):
                edge = (vertex_number, vertex_number+num_columns)
                strat_graph.add_edge(edge)

                if vertex_number not in strat_dict:
                    strat_dict[vertex_number] = set()
                strat_dict[vertex_number].add(edge)

                if vertex_number+num_columns not in strat_dict:
                    strat_dict[vertex_number+num_columns] = set()
                strat_dict[vertex_number+num_columns].add(edge)

            else:
                edge1 = (vertex_number, vertex_number+1)
                edge2 = (vertex_number, vertex_number+num_columns)
                strat_graph.add_edge(edge1)
                strat_graph.add_edge(edge2)

                if vertex_number not in strat_dict:
                    strat_dict[vertex_number] = set()
                strat_dict[vertex_number].add(edge1)
                strat_dict[vertex_number].add(edge2)

                if vertex_number+1 not in strat_dict:
                    strat_dict[vertex_number+1] = set()
                strat_dict[vertex_number+1].add(edge1)

                if vertex_number+num_columns not in strat_dict:
                    strat_dict[vertex_number+num_columns] = set()
                strat_dict[vertex_number+num_columns].add(edge2)

            vertex_number += 1

    return (strat_graph, strat_dict)

def build_edge_intersect_dict(strat_dict, num_columns, num_rows):
    edge_intersect_dict = dict()
    for vertex, edges in strat_dict.items():
        visited = set() 

        for edge in edges:
            if edge in visited:
                continue

            else:
                if (max(edge) - min(edge)) == 1:
                    depth = (max(edge) // num_columns)
                    coordinate1 = min(edge) + depth + 1
                    coordinate2 = max(edge) + num_columns + depth + 1
                  
                    edge_intersect_dict[edge] = (coordinate1, coordinate2)

                elif (max(edge) - min(edge)) > 1:
                    depth = (max(edge) // num_rows)
                    coordinate1 = min(edge) + num_rows + depth
                    coordinate2 = min(edge) + num_rows + depth + 1
                    edge_intersect_dict[edge] = (coordinate1, coordinate2)

                visited.add(edge)

    return edge_intersect_dict

def edge_to_spiel_move(game_dict, edge):
    first, second = edge
    for i in list((game_dict.keys())):
        if (game_dict[i] == first):
            first_num = i
        if (game_dict[i] == second):
            second_num = i
    return first_num, second_num

def spiel_move_to_edge(game_dict, spiel_move):
    first, second = spiel_move
    first_point = game_dict[first]
    second_point = game_dict[second]
    return first_point, second_point

def edge_to_spiel_action(edge, num_rows_1, num_cols_1):
    total_moves = num_rows_1*(num_cols_1 + 1) +  num_cols_1*(num_rows_1 + 1)
    (first_y, first_x), (second_y, second_x) = edge
    if (first_y == second_y):
        offset = 0
    else:
        offset = total_moves//2 + 1
    if (offset == 0):
        offset += first_y * num_rows_1
        offset += first_x
    else:
        if (first_y > 0):
            offset += (first_y) * num_cols_1 + 1
        offset += first_x
    return offset

def protocol_moveonly(serial_in, serial_out, all_moves, all_indices, move_for_computer, num_rows_1, num_columns_1):
    global game_over, error
    global game_graph, game_dict, box_dict
    global strat_graph, strat_dict, strat_box_dict, computer_is_first
    global edge_intersect_dict
    global num_columns, num_rows, num_moves
    global game_move, computer_move

    game_over = False
    error = False
    game_type = int()
    num_humans = 1
    num_columns = num_rows_1
    num_rows = num_columns_1

    (game_graph, game_dict, box_dict, strat_box_dict) = build_game_graph(num_columns, num_rows)

    (strat_graph, strat_dict) = build_strat_graph(game_dict, num_columns, num_rows)

    edge_intersect_dict = build_edge_intersect_dict(strat_dict, num_columns, num_rows)

    num_dots = ((num_columns + 1) * (num_rows + 1))
    num_boxes = (num_columns  * num_rows)
    num_moves = num_dots + num_boxes - 1

    computer_move = move_for_computer
    game_move = 1

    for i in range(0, len(all_moves)):
        game_move = all_indices[i]
        process_line(serial_in, serial_out, edge_to_spiel_move(game_dict, all_moves[i]))
    intelligent_agent_move, is_intelligent = computer_turn(serial_in, serial_out)
    return spiel_move_to_edge(game_dict, intelligent_agent_move), is_intelligent
