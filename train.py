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

    # zwei Agents – einer spielt X, einer O
    agent_X = QLearningAgent(epsilon=1.0)
    agent_O = QLearningAgent(epsilon=1.0)

    # Pretrain für beide Spieler
    if use_csv_data:
        pretrain_with_csv_data(agent_X, csv_data_file)
        pretrain_with_csv_data(agent_O, csv_data_file)

    print(f"=== Starting main RL training (AI vs AI) ===")

    for ep in range(episodes):
        board = env.reset()
        done = False

        while not done:
            # Agent auswählen je nach Spieler
            current_agent = agent_X if env.current_player == 1 else agent_O

            state_key = make_state_key(board, env.current_player)
            legal = env.legal_actions()
            action = current_agent.get_action(state_key, legal, training=True)

            # Zug ausführen
            next_board, reward, done, info = env.step(action)

            # Endzustand nach eigenem Zug
            if done:
                next_state_key = make_state_key(next_board, env.current_player)
                current_agent.update(state_key, action, reward, next_state_key, [], True)
                board = next_board
                break

            # Zug des anderen Agents (auch KI)
            other_agent = agent_O if env.current_player == -1 else agent_X

            opp_state_key = make_state_key(next_board, env.current_player)
            opp_legal = env.legal_actions()
            opp_action = other_agent.get_action(opp_state_key, opp_legal, training=True)

            after_opp_board, opp_reward, done, info = env.step(opp_action)

            # Rewards aus Sicht beider Agents korrekt weitergeben:
            if done:
                if env.winner == 1:
                    agent_X_reward = 1
                    agent_O_reward = -1
                elif env.winner == -1:
                    agent_X_reward = -1
                    agent_O_reward = 1
                else:
                    agent_X_reward = 0
                    agent_O_reward = 0

                # Updates für beide Agents
                next_state_key_X = make_state_key(after_opp_board, 1)
                next_state_key_O = make_state_key(after_opp_board, -1)

                agent_X.update(state_key, action, agent_X_reward, next_state_key_X, [], True)
                agent_O.update(opp_state_key, opp_action, agent_O_reward, next_state_key_O, [], True)

            else:
                next_state_key = make_state_key(after_opp_board, env.current_player)

                # partieller Reward
                current_agent.update(state_key, action, 0, next_state_key, env.legal_actions(), False)
                other_agent.update(opp_state_key, opp_action, 0, next_state_key, env.legal_actions(), False)

            board = after_opp_board

        # Epsilon-Decay für BEIDE Agents
        agent_X.epsilon = max(MIN_EPSILON, agent_X.epsilon * EPSILON_DECAY)
        agent_O.epsilon = max(MIN_EPSILON, agent_O.epsilon * EPSILON_DECAY)

        if ep % 5000 == 0:
            print(f"Episode {ep}/{episodes} | eps X={agent_X.epsilon:.3f} | eps O={agent_O.epsilon:.3f}")

    # Am Ende: beide Q-Tables speichern
    agent_X.save("qtable_X.pkl")
    agent_O.save("qtable_O.pkl")

    print("Training completed (AI vs AI).")
    return agent_X, agent_O


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TicTacToe RL agent')
    parser.add_argument('--episodes', type=int, default=EPISODES, help='Number of training episodes')
    parser.add_argument('--no-csv', action='store_true', help='Skip CSV pre-training')
    parser.add_argument('--csv-file', type=str, default='tic-tac-toe.data', help='Path to CSV data file')
    parser.add_argument('--output', type=str, default='qtable.pkl', help='Output path for trained model')
    
    args = parser.parse_args()
    
    train(episodes=args.episodes, save_path=args.output, use_csv_data=not args.no_csv, csv_data_file=args.csv_file)
