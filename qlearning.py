import logging

from typing import List

class QLearning():
    def __init__(self,
                 S: List[int],
                 A: List[int],
                 gamma: float,
                 Q,
                 logger: logging.Logger):
        self.S = S
        self.A = A
