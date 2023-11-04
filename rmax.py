import logging
import numpy as np

from mdp import MaximumLikelihoodMDP, MDP
from planner import Planner
from typing import List


class Rmax(MaximumLikelihoodMDP):
    """MDP representing R-MAX."""

    def __init__(self,
                 S: int,
                 A: int,
                 rho,
                 gamma: float,
                 planner: Planner,
                 m: int,
                 rmax: float,
                 logger_name: str = None):
        """Initializes the instance.

        Args:
          S: state space size. States assumed to be 1 -> S.
          A: action space size. actions assumed to be 1 -> A.
          rho: matrix of reward (i = state, j = action)
          gamma: discount
          planner: planning instance.
          m: count threshold
          rmax: maximum threshold
          logger_name: optional logging name.
        """

        super().__init__(gamma, S, A)
        self.rho = rho
        self.planner = planner
        self.m = m
        self.rmax = rmax
        if logger_name:
            self.logger = logging.getLogger(logger_name)

    def __log(self, msg: str, level = logging.INFO):
        """Wrapper around logging."""
        if self.logger:
            self.logger.log(level, msg)

    def backup(self, s: int) -> float:
        return np.max([self.lookahead(s, a) for a in self.actions()])

    def lookahead(self, s: int, a: int) -> float:
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
        utilities = [trans_prob(s_next)*self.get_utility(s_next) for s_next in self.states()]
        utility = r + self.gamma * np.sum(utilities)
        self.__log(f"Lookahead, U({s}, {a}) = {utility}")
        return utility

    def to_mdp(self) -> MDP:
        raise NotImplementedError("unimplemented")

    def update(self, s: int, a: int, r: float, next_s: int):
        s_index = self.state_index(s)
        a_index = self.action_index(a)
        next_s_index = self.state_index(next_s)
        self.add_count(s, a, next_s)
        self.rho[s_index, a_index] = self.rho[s_index, a_index] + r
        self.planner.update(self, s_index, a_index, r, next_s_index)


MaximumLikelihoodMDP.register(Rmax)
