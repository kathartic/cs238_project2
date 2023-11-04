import abc

from scipy.sparse import lil_matrix
from typing import Hashable, List

class MDP(abc.ABC):
    """Abstract class defining an MDP."""

    def __init__(self, S: List[Hashable], A: List[Hashable]):
        """Should not be called directly.

        Args:
          S: list of states. States must be hashable.
          A: list of actions. Actions must be hashable.
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
        self.U = lil_matrix((len(S), 1))

    def actions(self) -> List[Hashable]:
        """Returns actions for this MDP."""
        return self.__a_map.keys()

    def state_index(self, s: Hashable) -> int:
        """Returns the index of s in the internal map.

        Args:
          s: state.

        Returns:
          the representation of s as an int that can be indexed in the internal
          count matrix.
        """
        return self.__s_map[s]

    def states(self) -> List[Hashable]:
        """Returns list of states for the model."""
        return self.__s_map.keys()

    def action_index(self, a: Hashable) -> int:
        """Returns the index of a in the internal map.

        Args:
          a: action.

        Returns:
          the representation of a as an int that can be indexed in the internal
          count matrix.
        """
        return self.__a_map[a]

    def get_utility(self, s: Hashable) -> float:
        """Returns utility for state s."""

        s_index = self.state_index(s)
        return self.U[s_index][0]

    def set_utility(self, s: Hashable, utility: float):
        """Sets utility for state s."""

        s_index = self.state_index(s)
        self.U[s_index][0] = utility

    def row_index(self, s: Hashable, a: Hashable) -> int:
        """Returns the index of (s, a) in the internal counts map."""

        s_index = self.state_index(s)
        a_index = self.action_index(a)
        # states 2, actions 3
        # s0a0  0*3 + 0 = 0
        # s0a1  0*3 + 1 = 1
        # s0a2
        # s1a0  1*3 + 0 = 3
        return s_index*len(self.__a_map) + a_index

    @abc.abstractmethod
    def lookahead(self, s: Hashable, a: Hashable) -> float:
        """Returns utility of performing action a from state s.

        Uses Bellman equation.

        Args:
          s: current state
          a: current action

        Returns:
          Utility given current state s and taking action a.
        """
        pass

    @abc.abstractmethod
    def backup(self, s: Hashable) -> float:
        """Returns utility of optimal action from state s.

        Returns max of lookahead().

        Args:
          s: current state

        Returns:
          Utility of optimal action.
        """
        pass

    @abc.abstractmethod
    def update(self, s: Hashable, a: Hashable, r: float, next_s: Hashable):
        """Updates model based on given parameters.

        Args:
          s: current state
          a: action to take from s
          r: reward for taking action a in state s
          next_s: next state to go to.
        """
        pass


class Planner(abc.ABC):
    """Abstract class defining a planner."""
    @abc.abstractmethod
    def update(self, model: MDP, s: Hashable, a: Hashable, r: float, next_s: Hashable):
        """Mutates the utility of state s in model.

        Not all arguments may be used, depending on the implementation.

        Args:
          model: Model to update.
          s: The utility of this state will be updated.
          a: action to take from this state.
          r: reward given for taking action a from state s, and ending in state
            next_s.
          next_s: state ending up in after taking action a from state s.
        """
