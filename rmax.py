import numpy as np

from planner import MDP, Planner
from scipy.sparse import lil_matrix
from typing import Hashable, List

class Rmax(MDP):
    """MDP representing R-MAX."""

    def __init__(self,
                 S: List[Hashable],
                 A: List[Hashable],
                 gamma: float,
                 planner: Planner,
                 m: int,
                 rmax: float):
        """Initializes the instance.
        
        Args:
          S: list of states. State must be hashable.
          A: list of actions. State must be hashable.
          gamma: discount
          planner: planning instance.
          m: count threshold
          rmax: maximum threshold
        """

        super().__init__(S, A)
        # Maps state-action pairs, to potential next states.
        # Row format:
        # state 0 - action 0
        # state 0 - action 1
        # ...
        # state 1 - action 0
        # ...
        # state |S-1| - action |A-1|
        self.rho = lil_matrix((len(S), len(A)))
        self.gamma = gamma
        self.planner = planner
        self.m = m
        self.rmax = rmax


    def lookahead(self, s: Hashable, a: Hashable) -> float:
        s_index = self.state_index(s)
        a_index = self.action_index(a)
        i = self.row_index(s, a)
        n = np.sum(self.N[i])
        if n < self.m:
            return self.rmax / (1 - self.gamma)
        r = self.rho[s_index, a_index] / n
        trans_prob = lambda next_state : self.N[i, self.state_index(next_state)] / n
        utilities = [trans_prob(s_next)*self.get_utility(s_next) for s_next in self.__s_map.keys()]
        return r + self.gamma * np.sum(utilities)
    
    
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


MDP.register(Rmax)
