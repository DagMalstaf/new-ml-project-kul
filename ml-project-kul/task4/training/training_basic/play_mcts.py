import pyspiel
from task4.agent.mcts.mcts_agent_V0 import MCTSAgent  

def run_single_game_simulation(mcts_config):
    # Initialize the game and the agent
    game = pyspiel.load_game(mcts_config['game_name'])
    agent = MCTSAgent(game,
                      num_simulations=mcts_config['num_simulations'],
                      exploration_coefficient=mcts_config['exploration_coefficient'])

    # Start the game state
    state = game.new_initial_state()
    print('Started the game.')

    # Run the game loop
    while not state.is_terminal():
        print("Current state:\n", state)
        action = agent.get_action(state)
        state.apply_action(action)
        print(f"Action taken: {action}\n")

    # Display the final state and the game result
    print("Final state:\n", state)
    print("Game result:", state.returns())
