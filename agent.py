import random, pickle
from collections import defaultdict
from typing import Tuple, List

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.99, epsilon=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))

    def get_action(self, state: Tuple[int], legal: List[int], training=True) -> int:
        if training and random.random() < self.epsilon:
            return random.choice(legal)
        qvals = [(self.Q[state][a], a) for a in legal]
        max_q = max(qvals, key=lambda x: x[0])[0]
        best = [a for q,a in qvals if q == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state, next_legal, done):
        q_old = self.Q[state][action]
        target = reward
        if not done:
            best_next = max([self.Q[next_state][a] for a in next_legal]) if next_legal else 0
            target += self.gamma * best_next
        self.Q[state][action] += self.alpha * (target - q_old)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path):
        with open(path, "rb") as f:
            raw = pickle.load(f)
        self.Q = defaultdict(lambda: defaultdict(float), raw)
