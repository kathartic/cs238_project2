import numpy as np

from planner import MDP, Planner
from scipy.sparse import lil_matrix

class Rmax(MDP):
    """MDP representing R-MAX."""

    def __init__(self,
                 S: list,
                 A: list,
                 gamma: float,
                 planner: Planner,
                 m: int,
                 rmax: float):
        """Initializes the instance.
        
        Args:
          S: list of states. State must be hashable. Mapped as index.
          A: list of actions. State must be hashable. Mapped as index.
          gamma: discount
          planner: planning instance.
          m: count threshold
          rmax: maximum threshold
        """

        self.__s_map = {s : index for (index, s) in enumerate(S)}
        self.__a_map = {a : index for (index, a) in enumerate(A)}
        # Maps state-action pairs, to potential next states.
        # Row format:
        # state 0 - action 0
        # state 0 - action 1
        # ...
        # state 1 - action 0
        # ...
        # state |S-1| - action |A-1|
        self.N = lil_matrix((len(S) * len(A), len(S)))
        self.rho = lil_matrix((len(S), len(A)))
        self.gamma = gamma
        self.U = lil_matrix((len(S), 1))
        self.planner = planner
        self.m = m
        self.rmax = rmax

    def __state_index(self, s) -> int:
        """Returns the index of s in the internal map.
        
        Args:
          s: state.
        
        Returns:
          the representation of s as an int that can be indexed in the internal
          count matrix.
        """
        return self.__s_map[s]


    def __action_index(self, a) -> int:
        """Returns the index of a in the internal map.
        
        Args:
          a: action.
        
        Returns:
          the representation of a as an int that can be indexed in the internal
          count matrix.
        """
        return self.__a_map[a]
    
    def __row_index(self, s, a) -> int:
        """Returns the index of (s, a) in the internal counts map."""
        s_index = self.__state_index(s)
        a_index = self.__action_index(a)
        # states 2, actions 3
        # s0a0  0*3 + 0 = 0
        # s0a1  0*3 + 1 = 1
        # s0a2
        # s1a0  1*3 + 0 = 3
        return s_index*len(self.__a_map) + a_index


    def lookahead(self, s, a) -> float:
        s_index = self.__state_index(s)
        a_index = self.__action_index(a)
        i = self.__row_index(s, a)
        n = np.sum(self.N[i])
        if n < self.m:
            return self.rmax / (1 - self.gamma)
        r = self.rho[s_index, a_index] / n
        trans_prob = lambda next_state : self.N[i, next_state] / n
        utilities = [trans_prob(s_next)*self.U[s_next, 0] for s_next in self.__s_map.values()]
        return r + self.gamma * np.sum(utilities)
    
    
    def backup(self, s) -> float:
        return np.max([self.lookahead(s, a) for a in self.__a_map.keys()])


    def update(self, s, a, r, next_s):
        i = self.__row_index(s, a)
        s_index = self.__state_index(s)
        a_index = self.__action_index(a)
        next_s_index = self.__state_index(next_s)
        self.N[i, next_s_index] = self.N[i, next_s_index] + 1
        self.rho[s_index, a_index] = self.rho[s_index, a_index] + r
        self.planner.update(self, s_index, a_index, r, next_s_index)


MDP.register(Rmax) 