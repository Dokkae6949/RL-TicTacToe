from game import TicTacToe
from agent import QLearningAgent

def play(qtable_path="qtable.pkl"):
    agent = QLearningAgent()
    agent.load(qtable_path)
    env = TicTacToe()
    state = env.reset()
    print("Du spielst O. Eingabe: Zahl 0–8 (linke obere Ecke = 0, rechte untere = 8)")

    while True:
        env.render()
        if env.current_player == 1:
            action = agent.get_action(state, env.legal_actions(), training=False)
            print(f"Agent spielt: {action}")
            state, _, done, _ = env.step(action)
        else:
            move = None
            legal = env.legal_actions()
            while move not in legal:
                try:
                    move = int(input("Dein Zug: "))
                except:
                    move = None
            state, _, done, _ = env.step(move)

        if done:
            env.render()
            if env.winner == 1: print("Agent gewinnt!")
            elif env.winner == -1: print("Du gewinnst!")
            else: print("Unentschieden!")
            break

if __name__ == "__main__":
    play()
