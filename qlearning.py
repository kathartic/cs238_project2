import logging
import numpy as np

from typing import List

class QLearning():
    def __init__(self,
                 S: List[int],
                 A: List[int],
                 gamma: float,
                 Q,
                 alpha: float,
                 logger: logging.Logger):
        self.S = S
        self.A = A
        self.gamma = gamma
        self.Q = Q
        self.logger = logger
        self.alpha = alpha

    def update(self, s: int, a: int, r: float, next_s: int):
        discounted_max = self.gamma*np.max(self.Q[next_s, :])
        q_sa = self.Q[s, a]
        self.Q[s, a] = q_sa + self.alpha*(r + discounted_max - q_sa)


