# evaluate.py
from game import TicTacToe
from agent import QLearningAgent
import random

def make_state_key(board_tuple, current_player):
    return (board_tuple, current_player)

def evaluate(agent: QLearningAgent, episodes=1000):
    env = TicTacToe()
    results = {"win": 0, "draw": 0, "lose": 0}

    for _ in range(episodes):
        board = env.reset()
        done = False
        while not done:
            if env.current_player == 1:
                # agent move
                state_key = make_state_key(board, env.current_player)
                action = agent.get_action(state_key, env.legal_actions(), training=False)
                board, _, done, _ = env.step(action)
            else:
                # random opponent move
                action = random.choice(env.legal_actions())
                board, _, done, _ = env.step(action)
        # count result
        if env.winner == 1:
            results["win"] += 1
        elif env.winner == -1:
            results["lose"] += 1
        else:
            results["draw"] += 1

    total = sum(results.values())
    print(f"Evaluation over {total} games:")
    print(f"  Win rate:  {results['win']/total*100:.1f}%")
    print(f"  Draw rate: {results['draw']/total*100:.1f}%")
    print(f"  Lose rate: {results['lose']/total*100:.1f}%")
    return results

if __name__ == "__main__":
    agent = QLearningAgent()
    agent.load("qtable.pkl")
    evaluate(agent, 10000)
