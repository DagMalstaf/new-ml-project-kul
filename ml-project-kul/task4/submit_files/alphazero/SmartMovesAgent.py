def chain_is_open(chain, strat_box_dict):
    '''Checks if a chain is open (all boxes of the chain can be enclosed in
    consecutive moves).

    Arguments:
        chain (set): A set of vertices representing an open chain.

        strat_box_dict (dict): A dictionary that maps strategy graph vertices
            to the edges that surround it.

    Runtime:
        O(n) where n is the number of vertices i nthe strategy graph.

    Returns:
        bool: True if the chain is open, False otherwise.
    '''
    count_of_threes = int()
    count_of_ones = int()
    count_of_zeroes = int()

    # count the number of vertices that have zero, one, or three drawn
    # edges surrounding them.
    for vertex in chain:
        # A strat_box_dict entry with only 1 value means
        # that 3 edges have been drawn (removed from strat_box_dict).
        if len(strat_box_dict[vertex]) == 1:
            count_of_threes += 1
        if len(strat_box_dict[vertex]) == 3:
            count_of_ones += 1
        if len(strat_box_dict[vertex]) == 4:
            count_of_zeroes += 1

    # if this condition is satisfied then the chain is open
    if (count_of_threes - 1) >= (2*count_of_zeroes + count_of_ones):
        return True
    else:
        return False


def take_chain(chain, strat_box_dict):
    ''' Updates requested_edge to be an edge that will enclose at least one box
    in a long chain (3 or more boxes), allowing the computer to play until all
    boxes in the chain are enclosed. Taking long chains is the primary method
    of scoring points using the long chain rule.

    Arguments:
        chain (set): A set of vertices representing an open chain.

        strat_box_dict (dict): A dictionary that maps strategy graph vertices
            to the edges that surround it.

    Runtime:
        O(n) where n is the number of vertices in the strategy graph.

    Returns:
        requested_edge (tuple): The edge to be taken in the chain.

        vertex (int): The vertex that maps to the edge that was taken in
            strat_box_dict.
    '''
    for vertex in chain:
        # If a vertex has 3 edges surrounding it already, the last edge should
        # be taken so that the move completes the box and scores a point.
        if len(strat_box_dict[vertex]) == 1:
            possible_edge = list(strat_box_dict[vertex])
            # Choose the edge.
            requested_edge = possible_edge[0]
            # Let the user know what move is performed.
            # print("Taking chain")
            return (requested_edge, vertex)

    # Return None if an edge cannot be found. The AI will then move on to see
    # if other moves can be performed instead.
    return (None, None)

def open_chain(chain, strat_box_dict):
    '''Updates requested_edge to be a move that opens a short chain (2 boxes).
    The AI should only try to open a chain if it is not in control of the game.
    By the long chain rule, the AI should try and take a long chain in its
    last move. If it does not think it can do this, it should try to trick the
    player into taking an open short chain so that the user may have to open a
    long chain for it.

    Arguments:
        chain (set): A set of vertices representing a short chain component of
            the graph that the AI wants to open.

        strat_box_dict (dict): A dictionary that maps strategy graph vertices
            to the edges that surround it.

    Runtime:
        O(n) where n is the number of vertices in the strategy graph.

    Returns:
        requested_edge (tuple): The edge that the AI should take in order to
            open the short chain.
    '''
    for vertex in chain:
        # If the vertex has only two edges surrounding it, then one of the other
        # two edges must be taken to open this chain.
        if len(strat_box_dict[vertex]) == 2:
            possible_edges = list(strat_box_dict[vertex])
            # Arbitrarily choose the edge from the possible edges. Either one
            # will open the chain.
            requested_edge = possible_edges[0]
            # Let the user know what move is performed.
            print("Opening chain")
            return requested_edge
    # Return None if no edge can be found like this. The AI will then move on to
    # take a random edge instead.
    return None

def get_random_edge(game_graph, box_dict):
    '''Returns a random edge in the game graph that is a valid move.

    Arguments:
        game_graph (UndirectedAdjacencyGraph): A graph representation of the
            game board.

        box_dict (dict): A dictionary that maps vertices to the edges that
            surround it. As edges are taken in the game, edges are removes from
            the values of the dictionary.

    Runtime:
        O(n) where n is the number vertices in the game graph.

    Returns:
        chosen_edge (tuple): A valid edge that is yet to be taken in the game.
    '''
    from random import randint # Needed for the move to be pseudorandom

    possible_edges = set()
    # Iterate through all of the vertices of the game:
    for vertex in box_dict:
        # If not all of the edges have been taken surrounding a vertex:
        if len(box_dict[vertex]) > 0:
            # Add the untaken edges to the set of possible edges to take.
            possible_edges.update(box_dict[vertex])

    # Get a randomly generated index.
    edge_index = randint(0, len(possible_edges)-1)
    # Convert the set to a list.
    possible_edges = list(possible_edges)
    # Get the edge at the random index.
    chosen_edge = possible_edges[edge_index]

    # print("Random move") # Let the user know what type of move is performed.
    return chosen_edge # Return the edge.

import sys

"""
client-server messaging facility

This module provides the ability to send to and receive messages from a
client attached to a serial port. A message is a string of ascii
characters terminated with a new-line, as understood by the TextSerial
interface.

It also provides a diagnostic message tunnel capability.  Any message
coming from the client that begins with "D" is intercepted and printed
on stderr (asuming logging in on).

Combined with the dprintf library for the arduino, this enables the
client code to supply diagnostic information as if it was connected to
it's own stderr.

"""


def escape_nl(msg):
    """
    It's nice to know if we actually sent a complete line ending in
    \n, so escape it for display.
    """
    if msg != '' and msg[-1] == "\n":
        return msg[:-1] + "\\n"
    return msg


def send_msg_to_client(channel, msg):
    """
    Send a message to the client over channel, and log it if
    logging on.  msg should not end with a new line, as one will be
    added by the print.  Flush to ensure complete line is sent.
    """

    # print(msg, file=channel, flush=True)
    print(msg)
    

def receive_msg_from_client(channel):
    """
    Wait for a message from the client.  If a diagnostic 'D' type
    message comes in, intercept it, print it on stderr,  and wait
    for a proper one to arrive.

    The message is returned unchanged, terminating new line included.
    """

    while True:
        # msg = next(channel)
        msg = input()

        # If message begins with a D, then its a diagnostic message
        # from the client, and should be sent to stderr and ignored.

        if msg.strip()[:1] == "D":
            print(escape_nl(msg), file=sys.stderr, flush=True)
            continue
        else:
            break

    return msg

def coords_to_vertex(game_dict, coordinates):
    for vertex_identifier, vertex_coordinates in game_dict.items():
        # If the coordinates match any in the dictionary, return their vertex.
        if coordinates == vertex_coordinates:
            return vertex_identifier
    return -1 # Return -1 if no vertex is found.

def vertex_to_coords(game_dict, vertex):
    for vertex_identifier, vertex_coordinates in game_dict.items():
        # If the vertex matches any in the dictionary, return its coordinates.
        if vertex == vertex_identifier:
            return vertex_coordinates
    return -1 # Return -1 if no coordinates are found.

def draw_line(serial_in, serial_out, requested_edge):
    # game_graph is a graph representation of the game board. game_dict is a
    # dictionary that maps vertices to their x and y coordinates. box_dict is a
    # dictionary that maps game vertcies to the edges that box them in.
    global game_graph, game_dict, box_dict

    # strat_graph is a graph representation of the inner chains of the game
    # graph. edge_intersect_dict is a dictionary mapping strategy graph edges
    # tothe game graph edges that intersect them.
    global strat_graph, edge_intersect_dict

    # The number of total moves in the game, the present player turn (1 or 2),
    # and the turn that the computer plays on (1 or 2).
    global num_moves, game_move, computer_move

    # print("Reached draw_line")
    # print("game_graph edges", game_graph._vertices)
    # print("requested edge", requested_edge)
    # print("checking if already present : ", game_graph.is_edge(requested_edge))
    # If the line has already been drawn or coordinates are the same, do not
    # draw the line. The line request is invalid.
    if game_graph.is_edge(requested_edge) or \
        requested_edge[0] == requested_edge[1]:
        # print("Is already an edge in graph")
        # # tell client that the line request is invalid.
        # send_msg_to_client(serial_out, "L 1")

        # Get client acknowledgement. Return -1 if there is communication error.
        client_acknowledged(serial_in)
        if error: return -1

        return 0 # Return 0 because a line was not drawn.

    # If line has not been drawn before, process the drawing of the line.
    num_moves -= 1 # Decrement number of total moves
    game_graph.add_edge(requested_edge) # Add the edge to the game_graph
    # print("Edge has been added")
    # This added line may break a chain. If it does, remove the intersected edge
    # in the strat_graph.
    # Tuples have order so the reverse of the request needs to be checked.
    rev_requested_edge = (requested_edge[1], requested_edge[0])
    # print("Removing intersected Edges")
    for edge, intersecting_edge in edge_intersect_dict.items():
        # If the requested edge intersects a strategy graph edge:
        if requested_edge == intersecting_edge:
            # Remove the edge from the strategy graph
            strat_graph.remove_edge(edge)
            # Keep track that the strategy edge is now intersected.
            edge_intersect_dict[edge] = None

        # If the reverse requested edge interesects a strategy graph edge:
        elif rev_requested_edge == intersecting_edge:
            # Remove the edge from the strategy graph
            strat_graph.remove_edge(edge)
            # Keep track that the strategy edge is now interesected.
            edge_intersect_dict[edge] = None

    # # If the line is valid and it is a computer turn:
    # if computer_move == game_move:
    #     # Get the start vertex of the computer-chosen edge.
    #     start = vertex_to_coords(game_dict, min(requested_edge))

    #     # Send the x-coordinate of start vertex to the client.
    #     # send_msg_to_client(serial_out, "E {}".format(start[0]))
    #     # client_acknowledged(serial_in) # Get client acknowledgement.
    #     if error: return -1 # Return -1 if there is communication error.

    #     # Send the y-coordinate of the start vertex to the client.
    #     # send_msg_to_client(serial_out, "E {}".format(start[1]))
    #     # client_acknowledged(serial_in) # Get client acknowledgement.
    #     if error: return -1 # Return -1 if there is communication error.

    #     # Get the end vertex of the computer-chosen edge.
    #     end = vertex_to_coords(game_dict, max(requested_edge))

    #     # Send the x-coordinate of the end vertex to the client.
    #     # send_msg_to_client(serial_out, "E {}".format(end[0]))
    #     # client_acknowledged(serial_in) # Get client acknowledgement.
    #     if error: return -1 # Return -1 if there is communication error.

    #     # Send the y-coordinate of the end vertex to the client.
    #     # send_msg_to_client(serial_out, "E {}".format(end[1]))
    #     # client_acknowledged(serial_in) # Get client acknowledgement.
    #     if error: return -1 # Return -1 if there is communication error.

    # # The line is valid and it is a human turn:
    # else:
    #     # Tell client that a line was drawn
    #     # send_msg_to_client(serial_out, "L 0")
    #     # client_acknowledged(serial_in) # Get client acknowledgement.
    #     if error: return -1 # Return -1 if there is communication error.
    # print("Completed draw_line")
    return 1 # Return 1 because a line was drawn successfully.

def get_boxes(serial_in, serial_out, requested_edge):
    '''The requested_edge is removed from box_dict and strat_box_dict. If the
    edge is the last one needed to close one or two boxes, the information of
    these closed boxes is returned so that it may be sent to the client for
    drawing.

    Arguments:
        serial_in: Serial port input channel.

        serial_out: Serial port output channel.

        requested_edge (tuple): A requested game graph edge to draw.

    Runtime:
        O(n) where n is the number of vertices in the game graph.

    Returns:
        boxes (list): List of closed boxes (identified by an integer).
        len(boxes) (int): The number of closed boxes.
    '''
    # A dictionary mapping game vertices to the edges that box them in.
    global box_dict

    # A dictionary mapping strategy vertices to the edges that box them in.
    global strat_box_dict

    boxes = list() # A list of boxes to draw
    # Tuples have order so the reverse edge needs to be checked as well
    rev_requested_edge = (requested_edge[1], requested_edge[0])
    for box in box_dict:

        # If the requested edge is in the box dictionary, remove it.
        if requested_edge in box_dict[box]:
            box_dict[box].remove(requested_edge)

            # If the requested edge completes the box, put it in a list of
            # boxes to draw.
            if len(box_dict[box]) == 0:
                boxes.append(box)

        # Check the reverse requested edge similarly.
        elif rev_requested_edge in box_dict[box]:
            box_dict[box].remove(rev_requested_edge)

            # If the requested edge completes the box, put it in a list of
            # boxes to draw.
            if len(box_dict[box]) == 0:
                boxes.append(box)

    for box in strat_box_dict:
        # If the requested edge is in the box dictionary, remove it.
        if requested_edge in strat_box_dict[box]:
            strat_box_dict[box].remove(requested_edge)

        # Check the reverse requested edge similarly.
        elif rev_requested_edge in strat_box_dict[box]:
            strat_box_dict[box].remove(rev_requested_edge)

    # Return a list of boxes that were closed and the number of boxes that were
    # closed.
    return (boxes, len(boxes))

def process_line(serial_in, serial_out, requested_edge):
    '''Processes a requested edge by the human or computer. Calls draw_line to
    check if requested_edge is valid. Calls get_boxes to check if
    requested_edge causes any boxes to become enclosed. Send the information
    about closed boxes and whether the game is over to the client.

    Arguments:
        serial_in: Serial port input channel.

        serial_out: Serial port output channel.

        requested_edge (tuple): A requested game graph edge to draw.

    Runtime:
        O(n) where n is the number of closed boxes (max 2). However, calls
            draw_line and get_boxes which have their own runtimes.

    Returns:
        game_over (bool): Notifies whether the game is over.

        error (bool): Notifies whether there is a communication error.
    '''
    # game_over boolean notifies whether the game is over. error boolean
    # notifies whether there is a communication error.
    global game_over, error

    # Game dictionary mapping vertices to their coordinates.
    global game_dict

    # The current move number and the computer's move number
    global game_move, computer_move

    # print("Move : ", requested_edge)

    # Find out if a line was drawn (if the requested edge was a valid move)
    line_drawn = draw_line(serial_in, serial_out, requested_edge)
    # If a line was not drawn, return (game_over = False, error = False).
    if line_drawn < 1:
        return (game_over, error)
    # print("Getting Boxes")
    # Find out if and how many boxes were closed by the last move.
    (boxes, num_boxes) = get_boxes(serial_in, serial_out, requested_edge)
    # print("Boxes : ", boxes)
    # print("Num Boxes : ", num_boxes)

    # Send the number of closed boxes to the client.
    # send_msg_to_client(serial_out, "N {}".format(num_boxes))
    # If the client does not acknowledge, reset.
    # client_acknowledged(serial_in)
    # if error:
    #     return (game_over, error)

    # # Send the coordinates of every box to draw to the client.
    # for i in range(num_boxes):
    #     # Send the x-coordinate of the game vertex corresponding to the box to
    #     # draw to the client
    #     # send_msg_to_client(serial_out, "B {}"\
    #         # .format(vertex_to_coords(game_dict, boxes[i])[0]))

    #     # If the client does not acknowledge, reset.
    #     client_acknowledged(serial_in)
    #     if error: return (game_over, error)

    #     # Send the y-coordinate of the game vertex corresponding to the box to
    #     # draw to the client
    #     # send_msg_to_client(serial_out, "B {}"\
    #     #     .format(vertex_to_coords(game_dict, boxes[i])[1]))

    #     # If the client does not acknowledge, reset.
    #     client_acknowledged(serial_in)
    #     if error: return (game_over, error)

    # If no points were scored, the player turn is switched.
    # if num_boxes == 0:
    #     if game_move == 1:
    #         game_move = 2
    #     else:
    #         game_move = 1

    # If all possible moves have been played, the game is over.
    # if (num_moves == 0):
    #     game_over = True # The game is over.

    #     # Send that the game is over to the client.
    #     # send_msg_to_client(serial_out, "O 1")

    #     # The game will reset whether the client acknowledges or not.
    #     client_acknowledged(serial_in)
    #     # Client will send an extra 'A' to ensure that the player has clicked
    #     # the joystick to play again.
    #     client_acknowledged(serial_in)
    # else:
    #     game_over = False # The game is not over.

    #     # Send that the game is not over to the client.
    #     # send_msg_to_client(serial_out, "O 0")

    #     # If the client does not acknowledge, reset.
    #     client_acknowledged(serial_in)
    #     if error: return (game_over, error)
    # print("Completed Process Line")
    return (game_over, error)

def computer_turn(serial_in, serial_out):
    '''A computer turn uses the long chain rule to determine what move to
    determine what line to draw.

    Long chain rule:
        - If a chain of three or more boxes (long) can be scored, take them.
        - If a chain of two boxes (short) can be scored, only take these chains
            if the computer has control over the game. Control in dots and
            boxes is determined by the player's move order and the number of
            long chains on the board.
        - If the computer does not have control, attempt to trick the user into
            opening a new long chain by not taking a short chain and instead
            baiting the user by making the short chain closeable.
        - If there is not enough chain information, play a random move.

    Arguments:
        serial_in: Serial port input channel.

        serial_out: Serial port output channel.

    Runtime:
        O(n*(n+m)) where n is the number of vertices in the strat_graph and m
            is the number of edges in the strat_graph (bounded by the
            get_components function in traversal.py).

    Returns:
        An integer (-1, 0, or 1) depending on whether process_line returns an
            error, no line drawn, or that a line was drawn.
    '''
    # game_graph is a graph representation of the game board. box_dict is a
    # dictionary that maps game vertcies to the edges that box them in.
    global game_graph, box_dict

    # strat_graph is graph representation of the chains of connected boxes in
    # the game board. stored_chain is a chain that the computer may be in the
    # process of taking. edge_intersect_dict is dictionary mapping strategy
    # graph edges to the game graph edges that interesect them.
    # computer_is_first is whether the computer played first or not.
    global strat_graph, stored_chain, edge_intersect_dict, computer_is_first
    # strat_box_dict maps strategy graph vertices to the edges that box them in.
    global strat_box_dict

    # Number of game columns and rows.
    global num_columns, num_rows

    # Used to visualize the components that the AI is working with.
    # global debug

    # The number of dots in the game is used in addition to number of long
    # chains determining which player is in control.
    # print("inside computer_turn : ", game_graph._vertices)
    num_dots = ((num_columns + 1) * (num_rows + 1))

    # If a series of moves to score many boxes is present, finish the series and
    # score every box possible.
    if len(stored_chain) > 0:
        # Use take chain to get a suitable edge and a chosen vertex of the
        # stored chain.
        (requested_edge, chosen) = take_chain(stored_chain, strat_box_dict)
        # print("Taken Chain1 : ", (requested_edge, chosen))
        if not chosen is None:
            stored_chain.remove(chosen)
        else:
            stored_chain = list()
        # If the requeste edge is not None, process it.
        if not requested_edge is None:
            # print("Processing1 : ", requested_edge)
            process_line(serial_in, serial_out, requested_edge)
            return requested_edge, True

    # Get all the dijoint components of the strat_graph
    components = get_components(strat_graph)
    # if debug: print(components) # Visualize this
    # print("checkpoint1")

    # Create a list of subgraphs of the strat_graph
    subgraph_list = list()
    for vertex_set in components:
        # Each component is converted to a subgraph of the strat_graph
        subgraph = UndirectedAdjacencyGraph()

        # If there is a single vertex, do not factor into strategy.
        if len(vertex_set) == 1:
            continue

        # Add all of the vertices of the component into a new subgaph of the
        # main graph.
        for v in vertex_set:
            if not subgraph.is_vertex(v):
                subgraph.add_vertex(v)
        subgraph_list.append(subgraph)

        # If there is nothing obstructing the a strat_edge that relates to the
        # vertics of the subgaph, add the edge to the subgraph.
        for edge, intersect_edge in edge_intersect_dict.items():
            if not intersect_edge is None:
                if subgraph.is_vertex(edge[0]) and subgraph.is_vertex(edge[1]):
                    subgraph.add_edge(edge)
    # print("checkpoint2")

    long_chains = list() # A list of chains of 3 or more boxes
    short_chains = list() # A list of chain of 2 boxes
    for subgraph in subgraph_list:
        # If the subgraph is cyclic, it cannot be a chain.
        if subgraph.is_cyclic():
            # if debug: print("Cyclic:")
            # if debug: print(subgraph.vertices())
            continue
        else:
            # if debug: print("Not cyclic:")
            # if debug: print(subgraph.vertices())
            # Put long chains in one list, and short chains in another.
            if len(subgraph.vertices()) >= 3:
                long_chains.append({v for v in subgraph.vertices()})
            else:
                short_chains.append({v for v in subgraph.vertices()})
    # print("checkpoint3")
    # If a long chain has been opened, take it.
    if len(long_chains) > 0:
        # Sort the long chains from longest to shortest.
        sorted_long_chains = sorted(long_chains, key=len, reverse=True)

        # If there is more than one long chain, try to take the longest.
        for chain in sorted_long_chains:
            # If the chain is open, take it without question.
            if chain_is_open(chain, strat_box_dict):
                # Store the chain so that the AI takes all of it.
                stored_chain = chain
                # Score one of the boxes of the chain.
                (requested_edge, chosen) = \
                    take_chain(stored_chain, strat_box_dict)
                # print("Taken Chain2 : ", (requested_edge, chosen))

                # Remove the chosen vertex from the chain being taken.
                if not chosen is None:
                    stored_chain.remove(chosen)
                else:
                    stored_chain = list()
                # If the requested_edge is not None, process it for drawing.
                if not requested_edge is None:
                    # print("Processing2 : ", requested_edge)
                    process_line(serial_in, serial_out, requested_edge)
                    return requested_edge, True
    # print("checkpoint4")
    # If a long chain is not open, determine whether the computer has control
    # over the game.
    if (num_dots + len(long_chains)) % 2 == 0 and computer_is_first:
        # The computer is in control. It should wait for a long chain to be
        # opened.
        computer_has_control = True
    elif (num_dots + len(long_chains)) % 2 != 0 and not computer_is_first:
        # The computer is in control. It should wait for a long chain to be
        # opened.
        computer_has_control = True
    else:
        # The human has control. The computer needs to trick the human into
        # losing control by baiting short chains.
        computer_has_control = False
    # print("checkpoint5")
    # If the computer has control over the game, play on open short chains or
    # play a random edge that will not ruin control.
    if computer_has_control:
        for chain in short_chains:
            # If a short chain is open and the computer has control, there is
            # no problem with taking a short chain.
            if chain_is_open(chain, strat_box_dict):
                # Store the chain so that the AI takes all of it.
                stored_chain = chain
                # Score one of the boxes of the chain.
                (requested_edge, chosen) = \
                    take_chain(stored_chain, strat_box_dict)
                # print("Taken Chain3 : ", (requested_edge, chosen))

                # Remove the chosen vertex from the chain being taken.
                if not chosen is None:
                    stored_chain.remove(chosen)
                else:
                    stored_chain = list()
                # If the requested_edge is not None, process it for drawing.
                if not requested_edge is None:
                    # print("Processing3 : ", requested_edge)
                    process_line(serial_in, serial_out, requested_edge)
                    return requested_edge, False
    # If the computer does not have control, try to bait the user by
    # making a short chain closeable. The goal behind this is to make
    # the user have to play a move that makes a long chain closeable.
    else:
        for chain in short_chains:
            # Make sure the chain is not open, and then bait the player by
            # opening it.
            if not chain_is_open(chain, strat_box_dict):
                requested_edge = open_chain(stored_chain, strat_box_dict)
                # If the requested_edge is not None, process the edge for
                # drawing.
                if not requested_edge is None:
                    return process_line(serial_in, serial_out, requested_edge), True
    # print("checkpoint6")
    # If there are not suitable chains to play on, choose a random edge to play.
    # Guaranteed to return an edge.
    requested_edge = get_random_edge(game_graph, box_dict)
    # print("checkpoint7")
    # Process the random edge for drawing.
    # return process_line(serial_in, serial_out, requested_edge)
    # print("Completed Process_line", requested_edge)
    return requested_edge, False

def random_turn(serial_in, serial_out):
    requested_edge = get_random_edge(game_graph, box_dict)
    return requested_edge

def human_turn(serial_in, serial_out):
    '''A human turn relies on a request from the client. When a request is
    received, the edge is validated and this information is sent to the
    client using process_line.

    Arguments:
        serial_in: Serial port input channel.

        serial_out: Serial port output channel.

    Runtime:
        O(1) because a human turn only sends information to the client.

    Returns:
        An integer (-1, 0, or 1) depending on whether process_line returns an
            error, no line drawn, or that a line was drawn.
    '''
    # game_graph is a graph representation of the game board. game_dict is a
    # dictionary that maps vertices to their x and y coordinates. box_dict is a
    # dictionary that maps game vertcies to the edges that box them in.
    global game_graph, game_dict, box_dict

    # The number of total moves in the game.
    global num_moves

    # print("Awaiting Human Turn")
    # Get a request message from the client.
    msg = receive_msg_from_client(serial_in).split()
    # print("Msg : ", msg)
    log_msg(msg)

    # If the request is not of the form "R # # # #", then it is invalid.
    if len(msg) != 5 or msg[0] != 'R':
        print("Invalid request received.")
        return 0

    # Map the coordinates to their vertex.
    start_vertex = coords_to_vertex(game_dict, (int(msg[1]), int(msg[2])))
    # Tuples have order, so if the edge is -1, try the reverse tuple.
    if start_vertex == -1:
        start_vertex = coords_to_vertex(game_dict, (int(msg[2]), int(msg[1])))

    # Map the coordinates to their vertex.
    end_vertex = coords_to_vertex(game_dict, (int(msg[3]), int(msg[4])))
    # Tuples have order, so if the edge is -1, try the reverse tuple.
    if end_vertex == -1:
        end_vertex = coords_to_vertex(game_dict, (int(msg[4]), int(msg[3])))

    # The requested edge is stored as a tuple of the two integer vertices.
    requested_edge = (start_vertex, end_vertex)

    # Process the requested edge, ensuring it is not an invalid operation.
    return process_line(serial_in, serial_out, requested_edge)

def client_acknowledged(serial_in):
    '''A function to handle client acknowledgements. If an acknowledgement is
    not properly read, both the client and server should reset to the start
    of the game. Sets a global boolean ("error") based on whether there was an
    error in communication or not.
    '''
    global error # A global boolean notifying functions about errors.

    # Receive a message from the client.
    # msg = receive_msg_from_client(serial_in).rstrip()
    # log_msg(msg)

    # # If the server does receive proper acknowledgement:
    # if len(msg) > 1 or msg[0] != 'A':
    #     # There was a timeout if a 'T' is received.
    #     if msg[0] == 'T':
    #         print("Client took too long to respond.")
    #         print("Resetting...")

    #     # There was an unexpected character.
    #     else:
    #         print("Client sent unexpected character.")
    #         print("Client sent {}.".format(msg[0]))
    #         print("Resetting...")
    #     error = True

    # # Proper acknowledgement was received.
    # else:
    #     error = False
    error = False

class UndirectedAdjacencyGraph:
    '''Type to represent undirected graphs using adjacency storage.

    Attributes:
        _vertices (dict): A dictionary mapping a vertex to the other vertices
            that it is connected in an edge with.
    '''

    def __init__(self):
        self._vertices = dict()

    def add_vertex(self, v):
        ''' Adds a new vertex with identifier v to the graph.

        Arguments:
            v (int): The vertex identifier to be added.

        Raises:
            RuntimeError: If the vertex was already in the graph.
        '''
        if v not in self._vertices:
            self._vertices[v] = list()
        else:
            raise RuntimeError("Bad argument:"
                               " Vertex {} already in the graph".format(v))

    def is_vertex(self, v):
        '''Checks whether v is a vertex of the graph.

        Arguments:
            v (int): The vertex to be checked.

        Returns:
            bool: True if v is a vertex of the graph, False otherwise.
        '''
        return v in self._vertices

    def add_edge(self, e):
        ''' Adds edge e to the graph.

        Arguments:
            e (tuple): The edge to be added as a tuple. The edge goes from e[0]
                (an int) to e[1] (an int).

        Raises:
            RuntimeError: When one of the vertices in the edge is not a vertex
                in the graph.
        '''
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
        ''' Checks whether an edge e exists in the graph.

        Arguments:
            e (tuple): The edge to be checked. The edge goes from e[0] (an int)
                to e[1] (an int).

        Returns:
            bool: True if e is an edge of the graph, False otherwise.
        '''
        if (e[1] in self._vertices[e[0]]) or (e[0] in self._vertices[e[1]]):
            return True
        else:
            return False

    def remove_edge(self, e):
        ''' Removes an edge if it exists.

        Arguments:
            e (tuple): The edge to be removed. The edge goes from e[0]
                (an int) to e[1] (an int).
        '''
        if self.is_edge(e):
            self._vertices[e[0]].remove(e[1])
            self._vertices[e[1]].remove(e[0])

    def neighbours(self, v):
        '''Returns the list of vertices that are neighbours to v.

        Arguments:
            v (int): A vertex of the graph.
        '''
        return self._vertices[v]

    def vertices(self):
        '''Returns the set of all vertices in the graph.'''
        return set(self._vertices.keys())

    def clear(self):
        '''Method to clear the game graph for consecutive games played.'''
        self._vertices = dict()

    def is_cyclic_util(self, v, visited, parent):
        '''A recursive util for finding cycles. Used by the is_cyclic method.

        Arguments:
            v (int): A vertex of the strategy graph.

            visited (dict): A dictionary mapping verticess to whether they have
                been visited i nthe search or not.

            parent (int): The root vertex that started the search through the
                the neighbours of v.

        Runtime:
            O(n+m) where n is the number of vertices in the graph and m is the
                number of edges in the graph (this is dfs).

        Returns:
            bool: True if the graph has a cycle in it, False otherwise.
        '''
        visited[v] = True # The root vertex has now been visited.
        # Search all vertices that the root vertex is en edge with.
        for i in self.neighbours(v):
            # If the vertex has not been found yet, recurse.
            if visited[i] == False:
                if (self.is_cyclic_util(i, visited, v)):
                    return True

            # If a vertex finds an already found vertex that is not its parent,
            # there must be a cycle.
            elif parent != i:
                return True

        # If no cycles are detected, return False.
        return False

    def is_cyclic(self):
        '''Method to determine if graph is cyclic.

        Runtime:
            O(n+m) where n is the number of vertices in the graph and m is the
                number of edges in the graph (this is dfs).

        Returns:
            bool: True if the graph has a cycle in it, False otherwise.
        '''
        # Create a dictionary mapping vertices to whether they have been
        # visited yet.
        visited = {v:False for v in self.vertices()}

        for i in self.vertices():
            # If the vertex has not been visited:
            if visited[i] == False:
                # Determine whether the vertex loops to an already seen vertex.
                if (self.is_cyclic_util(i, visited, -1)) == True:
                    return True

        # If no cycles are detected, return False.
        return False

def breadth_first_search(g, v):
    '''Discovers all vertices in graph g reachable from vertex v and returns
    the search graph. Paths on the search graph are guaranteed to follow
    shortest paths from v.

    Arguments:
        g (UndirectedAdjacencyGraph): Graph to search in.

        v (int): Vertex of the graph where the search starts.

    Runtime:
        O(n + m) where n is the number of vertices in the graph and m is the
            number of edges in the graph.

    Returns:
        reached (frozenset): A frozenset of the discovered vertices in the
            graph.
    '''
    import queue

    todolist = queue.deque([v])  # todolist also stores "from where"
    reached = {v}
    # While todolist is not empty:
    while todolist:
        u = todolist.popleft()
        for w in g.neighbours(u):
            # If the vertex has not been reached yet:
            if w not in reached:
                reached.add(w)  # w has been reached
                todolist.append(w) # w is now in todolist

    return frozenset(reached)

def get_components(g):
    '''Finds and returns all of the components of graph g.

    Arguments:
        g (UndirectedAdjacencyGraph): The game graph from which components are
            counted.

    Runtime:
        O(n*(n+m)) where n is the number of vertices in the graph and m is the
            number of edges in the graph.

    Returns:
        component_set (set): List of frozensets that contain vertices reachable
            from each vertex of the graph.
    '''
    component_set = set()
    for v in g.vertices():
        # Conduct a breadth first search to find the component that v is in.
        component = breadth_first_search(g, v)

        # If the component has already been found, continue.
        if component in component_set:
            continue

        # Else, add the component to the set of all components of the graph.
        else:
            component_set.add(component)

    return component_set

import sys # Needed for stdin/stdout communication

# Game state variables
game_over = bool() # Keeps track of whether the game is over.
error = bool() # Keeps track of whether there is a communication error.

# Game graph information
game_graph = UndirectedAdjacencyGraph() # Graph representation of game board
# Dictionary mapping vertices to their x and y coordinates.
game_dict = dict()
# Dictioanry mapping vertices to the edges that surround them.
box_dict = dict()

# Strategy (AI) graph information
# Graph representation of connected box chains.
strat_graph = UndirectedAdjacencyGraph()
# Dictionary mapping strategy vertices to the edges that they are a part of.
strat_dict = dict()
# Dictionary mapping strategy vertices to the edges that box them in.
strat_box_dict = dict()
# Keeps track of a chain of boxes that the AI is in the process of taking.
stored_chain = list()
# Keeps track of whether the computer played first in the game.
computer_is_first = bool()

# Game dimension variables
num_columns = int() # Number of columns in the game.
num_rows = int() # Number of rows in the game.
num_moves = int() # Number of total moves in the game.

# Variables to determine whose move it is
game_move = int() # The current turn (1 or 2)
computer_move = int() # The turn that the computer moves on (1 or 2)
def build_game_graph(num_columns, num_rows):
    '''A function that builds the game graph, a graphical representation of the
    dots and edges that are drawn to the Arduino screen. To assist in drawing
    capabilities and AI strategy, dictionaries store knowledge about vertices
    and edges in the graph.

    Arguments:
        num_columns (int): The number of columns that the game board has.

        num_rows (int): The number of rows that the game board has.

    Runtime:
        O(n*m) where n is the number of columns of the game board and m is the
            number of rows of the game board.

    Returns:
        game_graph (UndirectedAdjacencyGraph): An UndirectedAdjacencyGraph with
            vertices that correspond to the dots of the game board. Will be
            used to keep track of lines (edges) drawn throughout game. Vertices
            are labelled starting from 0 increasing left to right, top to
            bottom.

        game_dict (dict): A dictionary that maps vertices of game_graph to
            their coordinate position (x, y) on the game board.

        box_dict (dict): A dictionary that maps the the top left corner of a
            box to the edges that make up the box.

        strat_box_dict (UndirectedAdjacencyGraph): A dictionary that maps the
            vertices of strat_graph to the edges that surround it.
    '''
    # The game graph is the board seenand interacted with by the players.
    game_graph = UndirectedAdjacencyGraph()
    # Game dict maps vertices to their coordinates.
    game_dict = dict()
    vertex_number = int()
    for i in range(num_rows+1):
        for j in range(num_columns+1):
            # Add the vertex to the graph.
            game_graph.add_vertex(vertex_number)
            # Add map the vertex to its coordinates.
            game_dict[vertex_number] = (j, i)
            vertex_number += 1

    # box_dict is a dict that maps the position of the top left
    # corner of a box to the edges that make up the box.
    box_dict = dict()
    # strat_box_dict maps vertices in the strat_graph to the game_graph edges
    # that surround it (used by the AI).
    strat_box_dict = dict()
    # Vertex numbering related to the strat_graph
    strat_vertex = int()
    # Vertex numbering related to the game_graph
    game_vertex = int()
    for i in range(num_rows):
        for j in range(num_columns):

            # Top edge of box
            edge1 = (game_vertex, game_vertex+1)
            # Left edge of box
            edge2 = (game_vertex, game_vertex+num_columns+1)
            # Right edge of box
            edge3 = (game_vertex+1, game_vertex+num_columns+2)
            # Bottom edge of box
            edge4 = (game_vertex+num_columns+1, game_vertex+num_columns+2)

            box_dict[game_vertex] = {edge1, edge2, edge3, edge4}
            strat_box_dict[strat_vertex] = {edge1, edge2, edge3, edge4}

            # There is one strategy vertex for every box. There is one game
            # vertex for every dot in the game board.
            strat_vertex += 1
            if j == num_columns-1:
                game_vertex += 2
            else:
                game_vertex += 1

    return (game_graph, game_dict, box_dict, strat_box_dict)

def build_strat_graph(game_dict, num_columns, num_rows):
    '''A function that builds the AI strategy graph, a graphical representation
    of the chains that link columns and rows within the graph. A strategy
    dictionary is also built to assist in the building of another dictionary
    used in AI strategy.

    Arguments:
        game_dict (dict): A dictionary that maps vertices to their x and y
            coordinates.

        num_columns (int): The number of columns that the game board has.

        num_rows (int): The number of rows that the game board has.

    Runtime:
        O(n*m) where n is the number of columns of the game board and m is the
            number of rows in the game board.

    Returns:
        strat_graph (UndirectedAdjacencyGraph): An UndirectedAdjacencyGraph
            with vertices that correspond to the centers of the boxes in the
            game. The initial edges connect all the vertices in a grid pattern.
            Is used to keep track of long chains for AI strategy.

        strat_dict (dict): A dictionary that maps the vertices of strat_graph
            to the edges that they are a part of. Used to build
            edge_intersect_dict.
    '''
    strat_graph = UndirectedAdjacencyGraph()

    # Add vertices to strat_graph.
    for v in range(num_columns*num_rows):
        strat_graph.add_vertex(v)

    # Add strat_graph edges and build strat_dict which maps strat_graph vertices
    # to the edges that they are a part of.
    strat_dict = dict()
    vertex_number = int()
    for i in range(num_rows):
        for j in range(num_columns):
            # strat_graph is built from top-left to bottom-right. Therefore,
            # the bottom-right vertex should not add any edges.
            if i == (num_rows - 1) and j == (num_columns - 1):
                pass

            # Bottom vertices should only add an edge to their right.
            elif i == (num_rows - 1):
                # Add the edge to the strategy graph.
                edge = (vertex_number, vertex_number+1)
                strat_graph.add_edge(edge)

                # Add the edge to the set of edges mapped to by the vertex
                if vertex_number not in strat_dict:
                    strat_dict[vertex_number] = set()
                strat_dict[vertex_number].add(edge)

                # Add the edge to the set of edges mapped to by the vertex
                if vertex_number+1 not in strat_dict:
                    strat_dict[vertex_number+1] = set()
                strat_dict[vertex_number+1].add(edge)

            # Right vertices should only add an edge below them.
            elif j == (num_columns - 1):
                # Add the edge to the strategy graph.
                edge = (vertex_number, vertex_number+num_columns)
                strat_graph.add_edge(edge)

                # Add the edge to the set of edges mapped to by the vertex
                if vertex_number not in strat_dict:
                    strat_dict[vertex_number] = set()
                strat_dict[vertex_number].add(edge)

                # Add the edge to the set of edges mapped to by the vertex
                if vertex_number+num_columns not in strat_dict:
                    strat_dict[vertex_number+num_columns] = set()
                strat_dict[vertex_number+num_columns].add(edge)

            # Vertices that are not on the right or bottom edges will add edges
            # to their right and below them.
            else:
                # Add the edges to the strategy graph.
                edge1 = (vertex_number, vertex_number+1)
                edge2 = (vertex_number, vertex_number+num_columns)
                strat_graph.add_edge(edge1)
                strat_graph.add_edge(edge2)

                # Add both edges to the set of edges mappes to by the vertex
                if vertex_number not in strat_dict:
                    strat_dict[vertex_number] = set()
                strat_dict[vertex_number].add(edge1)
                strat_dict[vertex_number].add(edge2)

                # Add the edge to the set of edges mapped to by vertex
                if vertex_number+1 not in strat_dict:
                    strat_dict[vertex_number+1] = set()
                strat_dict[vertex_number+1].add(edge1)

                # Add the edge to the set of edges mapped to by vertex
                if vertex_number+num_columns not in strat_dict:
                    strat_dict[vertex_number+num_columns] = set()
                strat_dict[vertex_number+num_columns].add(edge2)

            vertex_number += 1 # Increment the vertex number

    return (strat_graph, strat_dict)

def build_edge_intersect_dict(strat_dict, num_columns, num_rows):
    '''A function that builds a dictionary that maps strat_graph edges to
    game_graph edges that intersect them. This dictionary helps the AI determine
    what components of the graph are connected. If a drawn line interesects a
    certain chain of strat_graph vertices, the chain may go from a long chain
    to a short chain.

    Arguments:
        strat_dict (dict): A dictionary that maps strat_graph vertices to the
            edges that they are a part of.

        num_columns (int): The number of columns in the game board.

        num_rows (int): The number of rows in the game board.

    Runtime:
        O(n*m) where n is the number of vertices in the strat_graph and m is
            the number edges that the vertex belongs to (at most 4).

    Returns:
        edge_intersect_dict (dict): A dictionary that maps the initial edges of
            strat_graph to the edges in game_graph that intersect them. Used
            for removing edges from strat_graph when lines are drawn that
            intersect them.
    '''
    edge_intersect_dict = dict()
    # Iterate through the vertices of strat_graph
    for vertex, edges in strat_dict.items():
        visited = set() # Keep track of edges that have been seen already

        # A vertex can be a part of at most 4 edges.
        for edge in edges:
            # If the edge has been visited already, contnue.
            if edge in visited:
                continue

            # If the edge is new, map it to a game_graph edge that, when drawn,
            # will interesect it.
            else:
                # If the strat_graph edge is horizontal:
                if (max(edge) - min(edge)) == 1:
                    # Depth accounts for disparity between game vertex numbering
                    # and strat vertex numbering
                    depth = (max(edge) // num_columns)
                    coordinate1 = min(edge) + depth + 1
                    coordinate2 = max(edge) + num_columns + depth + 1

                    # Map a strat edge to the game edge that will interesect it
                    # when it is drawn.
                    edge_intersect_dict[edge] = (coordinate1, coordinate2)

                # if the strat_graph edge is vertical:
                elif (max(edge) - min(edge)) > 1:
                    # Depth accounts for disparity between game vertex numbering
                    # and strat vertex numbering
                    depth = (max(edge) // num_rows)
                    coordinate1 = min(edge) + num_rows + depth
                    coordinate2 = min(edge) + num_rows + depth + 1
                    # Map a strat edge to the game edge that will interesect it
                    # when it is drawn.
                    edge_intersect_dict[edge] = (coordinate1, coordinate2)

                # The strat edge has now been visited
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

def edge_to_spiel_action(edge):
    (first_y, first_x), (second_y, second_x) = edge
    if (first_y == second_y):
        offset = 0
    else:
        offset = 56
    if (offset == 0):
        offset += first_y * 7
        offset += first_x
    else:
        offset += first_x * 7
        offset += first_y
    return offset

def protocol_moveonly(serial_in, serial_out, all_moves, all_indices, move_for_computer):
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
    num_columns = 7
    num_rows = 7

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
