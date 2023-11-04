import logging
import numpy as np

from mdp import MaximumLikelihoodMDP
from planner import Planner
from scipy.sparse import lil_matrix
from typing import Hashable, List

class Rmax(MaximumLikelihoodMDP):
    """MDP representing R-MAX."""

    def __init__(self,
                 S: List[Hashable],
                 A: List[Hashable],
                 gamma: float,
                 planner: Planner,
                 m: int,
                 rmax: float,
                 logger_name: str = None):
        """Initializes the instance.

        Args:
          S: list of states. State must be hashable.
          A: list of actions. State must be hashable.
          gamma: discount
          planner: planning instance.
          m: count threshold
          rmax: maximum threshold
          logger_name: optional logging name.
        """

        super().__init__(gamma, S, A)
        self.rho = lil_matrix((len(S), len(A)))
        self.planner = planner
        self.m = m
        self.rmax = rmax
        if logger_name:
            self.logger = logging.getLogger(logger_name)

    def __log(self, msg: str, level = logging.INFO):
        """Wrapper around logging."""
        if self.logger:
            self.logger.log(level, msg)

    def lookahead(self, s: Hashable, a: Hashable) -> float:
        s_index = self.state_index(s)
        a_index = self.action_index(a)
        i = self.row_index(s, a)
        n = np.sum(self.N[i, :])
        self.__log(f"Lookahead, N({s}, {a}) = {n}")
        if n < self.m:
            utility =  self.rmax / (1 - self.gamma)
            self.__log(f"Lookahead, max discounted utility: U({s}, {a}) = {utility}")
            return utility

        r = self.rho[s_index, a_index] / n
        trans_prob = lambda next_state : self.N[i, self.state_index(next_state)] / n
        utilities = [trans_prob(s_next)*self.get_utility(s_next) for s_next in self.__s_map.keys()]
        utility = r + self.gamma * np.sum(utilities)
        self.__log(f"Lookahead, U({s}, {a}) = {utility}")
        return utility

    def backup(self, s: Hashable) -> float:
        return np.max([self.lookahead(s, a) for a in self.__a_map.keys()])

    def update(self, s: Hashable, a: Hashable, r: float, next_s: Hashable):
        i = self.row_index(s, a)
        s_index = self.state_index(s)
        a_index = self.action_index(a)
        next_s_index = self.state_index(next_s)
        self.N[i, next_s_index] = self.N[i, next_s_index] + 1
        self.rho[s_index, a_index] = self.rho[s_index, a_index] + r
        self.planner.update(self, s_index, a_index, r, next_s_index)


MaximumLikelihoodMDP.register(Rmax)
