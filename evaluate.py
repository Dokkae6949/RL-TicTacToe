# evaluate.py
from game import TicTacToe
from agent import QLearningAgent
from utils import make_state_key
import random

def evaluate(agent_X: QLearningAgent, agent_O: QLearningAgent, episodes=1000):
    env = TicTacToe()
    results = {"win": 0, "draw": 0, "lose": 0}

    for _ in range(episodes):
        board = env.reset()
        done = False

        while not done:
            # Agent X (player +1)
            if env.current_player == 1:
                state_key = make_state_key(board, 1)
                legal = env.legal_actions()
                action = agent_X.get_action(state_key, legal, training=False)
                board, _, done, _ = env.step(action)

            # Agent O (player -1)
            else:
                state_key = make_state_key(board, -1)
                legal = env.legal_actions()
                action = agent_O.get_action(state_key, legal, training=False)
                board, _, done, _ = env.step(action)

        # Count results from X's perspective
        if env.winner == 1:
            results["win"] += 1       # X wins
        elif env.winner == -1:
            results["lose"] += 1      # X loses
        else:
            results["draw"] += 1

    # Print summary
    total = sum(results.values())
    print(f"Evaluation over {total} AI-vs-AI games:")
    print(f"  X Win rate:  {results['win']/total*100:.1f}%")
    print(f"  Draw rate:   {results['draw']/total*100:.1f}%")
    print(f"  X Lose rate: {results['lose']/total*100:.1f}%")

    return results


if __name__ == "__main__":
    agent_X = QLearningAgent()
    agent_O = QLearningAgent()
    agent_X.load("qtable.pkl")
    agent_O.load("qtable.pkl")
    evaluate(agent_X, agent_O, 10000)
