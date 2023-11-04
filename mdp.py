import abc
import numpy as np

from scipy.sparse import lil_matrix
from typing import Hashable, List, Tuple


class MDP:
    """Defines a MDP."""

    def __init__(self,
                 gamma: float,
                 S: List[Hashable],
                 A: List[Hashable],
                 T,
                 R):
        """Instantiates an instance.

        Args:
          gamma: discount factor.
          S: State space.
          A: Action space.
          T: Transition matrix (i = state, j = action)
          R: Reward matrix (i = state, j = action).
        """
        self.gamma = gamma
        self.__s_map = {s : index for (index, s) in enumerate(S)}
        self.__a_map = {a : index for (index, a) in enumerate(A)}
        self.T = T
        self.R = R

    def __action_index(self, a: Hashable) -> int:
        """Returns the index of a in the internal map.

        Args:
          a: action.

        Returns:
          the representation of a as an int that can be indexed in the internal
          count matrix.
        """
        return self.__a_map[a]

    def __state_index(self, s: Hashable) -> int:
        """Returns the index of s in the internal map.

        Args:
          s: state.

        Returns:
          the representation of s as an int that can be indexed in the internal
          count matrix.
        """
        return self.__s_map[s]

    def __tr(self, s: Hashable, a: Hashable) -> Tuple[Hashable, float]:
        """Samples transition and reward.

        Args:
          s: state to take action from.
          a: action to take.

        Returns:
          Tuple of next state, reward.
        """
        s_index = self.__state_index(s)
        a_index = self.__action_index(a)
        next_state = np.random.choice(self.__s_map.keys(), p = self.T[s_index, a_index])
        return (next_state, self.R[s_index, a_index])

    def simulate(self, s: Hashable, policy) -> Tuple[Hashable, Hashable, float, Hashable]:
        """Simulates one policy rollout."""

        a = policy(s)
        next_s, r = self.__tr(s, a)
        return (s, a, r, next_s)


class MaximumLikelihoodMDP(abc.ABC):
    """Abstract class defining an MLE MDP."""

    def __init__(self, gamma: float, S: List[Hashable], A: List[Hashable]):
        """Should not be called directly.

        Args:
          gamma: discount factor.
          S: State space.
          A: Action space.
        """
        self.gamma = gamma
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
