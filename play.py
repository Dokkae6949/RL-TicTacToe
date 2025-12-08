from game import TicTacToe
from agent import QLearningAgent
from utils import make_state_key

def play(qtable_path="qtable.pkl"):
    agent = QLearningAgent()
    agent.load(qtable_path)
    env = TicTacToe()
    board = env.reset()
    print("You play O. Input: number 0–8 (top-left = 0, bottom-right = 8)")

    while True:
        env.render()
        if env.current_player == 1:
            state_key = make_state_key(board, env.current_player)
            action = agent.get_action(state_key, env.legal_actions(), training=False)
            print(f"Agent plays: {action}")
            board, _, done, _ = env.step(action)
        else:
            move = None
            legal = env.legal_actions()
            while move not in legal:
                try:
                    move = int(input("Your move: "))
                except:
                    move = None
            board, _, done, _ = env.step(move)

        if done:
            env.render()
            if env.winner == 1: print("Agent wins!")
            elif env.winner == -1: print("You win!")
            else: print("Draw!")
            break

if __name__ == "__main__":
    play()
