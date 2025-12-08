from game import TicTacToe
from agent import QLearningAgent
from data_loader import load_tictactoe_data
from utils import make_state_key
import random

EPSILON_DECAY = 0.99995
MIN_EPSILON = 0.05
EPISODES = 50000
CSV_PRETRAIN_EPISODES = 5000

def pretrain_with_csv_data(agent: QLearningAgent, data_file="tic-tac-toe.data", episodes=CSV_PRETRAIN_EPISODES):
    """
    Pre-train the agent using game states from CSV data.
    This helps bootstrap the Q-learning with knowledge from completed games.
    """
    print(f"\n=== Pre-training with CSV data ===")
    data = load_tictactoe_data(data_file)
    
    if not data:
        print("No CSV data available, skipping pre-training.")
        return
    
    env = TicTacToe()
    
    for ep in range(episodes):
        # Sample a random game state from the dataset
        board_state, outcome = random.choice(data)
        
        # Create a partial game state by removing some moves
        # This gives us non-terminal states to learn from
        board_list = list(board_state)
        filled_positions = [i for i, v in enumerate(board_list) if v != 0]
        
        if len(filled_positions) > 2:
            # Remove 1-3 moves to create a non-terminal state
            num_to_remove = random.randint(1, min(3, len(filled_positions) - 2))
            positions_to_clear = random.sample(filled_positions, num_to_remove)
            for pos in positions_to_clear:
                board_list[pos] = 0
        
        # Set up the environment with this partial state
        env.reset()
        env.board = board_list
        
        # Determine whose turn it is based on the number of pieces
        x_count = sum(1 for cell in board_list if cell == 1)
        o_count = sum(1 for cell in board_list if cell == -1)
        env.current_player = 1 if x_count <= o_count else -1
        
        # Check if game is done
        env._check_done()
        
        # Get legal actions
        legal = env.legal_actions()
        
        if legal and not env.done:
            # Let agent learn from this state
            state_key = make_state_key(tuple(env.board), env.current_player)
            action = agent.get_action(state_key, legal, training=True)
            
            # Simulate the action and learn
            next_board, reward, done, _ = env.step(action)
            
            # Adjust reward based on original outcome
            if done:
                if outcome == 'win' and env.winner == 1:
                    reward = 1.0
                elif outcome == 'negative' and env.winner == -1:
                    reward = -1.0
                elif env.winner == 0:
                    reward = 0.0
            else:
                reward = 0.0
            
            next_state_key = make_state_key(tuple(env.board), env.current_player)
            agent.update(state_key, action, reward, next_state_key, env.legal_actions(), done)
        
        if ep % 1000 == 0:
            print(f"  Pre-training episode {ep}/{episodes}, epsilon={agent.epsilon:.3f}")
    
    print(f"Pre-training completed with {len(data)} unique game states.\n")


def train(episodes=EPISODES, save_path="qtable.pkl", use_csv_data=True, csv_data_file="tic-tac-toe.data"):
    env = TicTacToe()
    agent = QLearningAgent(epsilon=1.0)
    
    # Pre-train with CSV data if available
    if use_csv_data:
        pretrain_with_csv_data(agent, csv_data_file)

    print(f"=== Starting main RL training ===")
    for ep in range(episodes):
        board = env.reset()
        done = False
        while not done:
            state_key = make_state_key(board, env.current_player)
            legal = env.legal_actions()
            action = agent.get_action(state_key, legal, training=True)

            # agent plays
            after_agent_board, reward_agent_move, done, info = env.step(action)

            if done:
                # terminal immediately after agent move
                next_state_key = make_state_key(after_agent_board, env.current_player)
                agent.update(state_key, action, reward_agent_move, next_state_key, [], True)
                board = after_agent_board
                break

            # opponent (random) plays
            opp_action = random.choice(env.legal_actions())
            after_opp_board, reward_opp_move, done, info = env.step(opp_action)

            # compute agent-centric reward AFTER opponent move
            if done:
                # game ended because of opponent move or draw
                if env.winner == 1:
                    final_reward = 1
                elif env.winner == -1:
                    final_reward = -1
                else:
                    final_reward = 0
            else:
                final_reward = 0

            next_state_key = make_state_key(after_opp_board, env.current_player)
            agent.update(state_key, action, final_reward, next_state_key, env.legal_actions(), done)
            board = after_opp_board

        # decay epsilon slowly
        agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)
        if ep % 5000 == 0:
            print(f"Episode {ep}/{episodes}, epsilon={agent.epsilon:.3f}")

    agent.save(save_path)
    print(f"Training completed and saved to {save_path}")
    return agent

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TicTacToe RL agent')
    parser.add_argument('--episodes', type=int, default=EPISODES, help='Number of training episodes')
    parser.add_argument('--no-csv', action='store_true', help='Skip CSV pre-training')
    parser.add_argument('--csv-file', type=str, default='tic-tac-toe.data', help='Path to CSV data file')
    parser.add_argument('--output', type=str, default='qtable.pkl', help='Output path for trained model')
    
    args = parser.parse_args()
    
    train(episodes=args.episodes, save_path=args.output, use_csv_data=not args.no_csv, csv_data_file=args.csv_file)
