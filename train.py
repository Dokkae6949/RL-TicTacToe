from game import TicTacToe
from agent import QLearningAgent
import random

def train(episodes=50000, save_path="qtable.pkl"):
    env = TicTacToe()
    agent = QLearningAgent(epsilon=1.0)

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            legal = env.legal_actions()
            action = agent.get_action(state, legal)
            next_state, reward, done, _ = env.step(action)

            # Gegner macht zufälligen Zug
            if not done:
                opp_action = random.choice(env.legal_actions())
                next_state, reward, done, _ = env.step(opp_action)

            agent.update(state, action, reward, next_state, env.legal_actions(), done)
            state = next_state

        # Exploration langsam reduzieren
        agent.epsilon = max(0.05, agent.epsilon * 0.99995)
        if ep % 5000 == 0:
            print(f"Episode {ep}/{episodes}, epsilon={agent.epsilon:.3f}")

    agent.save(save_path)
    print(f"Training abgeschlossen! Gespeichert unter {save_path}")
    return agent

if __name__ == "__main__":
    train()
