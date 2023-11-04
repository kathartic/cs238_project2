import logging
import numpy as np

from scipy.sparse import lil_matrix
from typing import List, Tuple


class MDP:
    """Defines a MDP."""

    def __init__(self,
                 gamma: float,
                 S: int,
                 A: int,
                 T,
                 R):
        """Instantiates an instance.

        Args:
          gamma: discount factor.
          S: Size of state space. Assumes states are 1 to S, inclusive.
          A: Size of action space. Assumes states are 1 to S, inclusive.
          T: Transition matrix (i = state, j = action)
          R: Reward matrix (i = state, j = action).
        """
        self.gamma = gamma
        self.state_count = S
        self.action_count = A
        self.T = T
        self.R = R

    def __action_index(self, a: int) -> int:
        """Returns the index of a in the internal map.

        Args:
          a: action.

        Returns:
          the representation of a as an int that can be indexed in the internal
          count matrix.
        """
        return a - 1

    def __state_index(self, s: int) -> int:
        """Returns the index of s in the internal map.

        Args:
          s: state.

        Returns:
          the representation of s as an int that can be indexed in the internal
          count matrix.
        """
        return s - 1

    def __tr(self, s: int, a: int) -> Tuple[int, float]:
        """Samples transition and reward.

        Args:
          s: state to take action from.
          a: action to take.

        Returns:
          Tuple of next state, reward.
        """
        s_index = self.__state_index(s)
        a_index = self.__action_index(a)
        next_state = np.random.choice(np.arange(1, self.state_count + 1),
                                      p = self.T[s_index, a_index])
        return (next_state, self.R[s_index, a_index])

    def simulate(self, s: int, policy) -> Tuple[int, int, float, int]:
        """Simulates one policy rollout."""

        a = policy(s)
        next_s, r = self.__tr(s, a)
        return (s, a, r, next_s)


class MaximumLikelihoodMDP():
    """Class defining an MLE MDP."""

    def __init__(self, S: int, A: int, gamma: float, planner, logger_name: str = None):
        """Instantiates a new instance.

        Args:
          S: Size of state space. States are assumed 1 - S, inclusive.
          A: Size of action space. States are assumed 1 - A, inclusive.
          gamma: discount factor.
          planner: Planner.
        """
        self.gamma = gamma
        self.state_space_size = S
        self.action_space_size = A
        self.rho = lil_matrix((S, A))
        self.planner = planner
        # Maps state-action pairs, to potential next states.
        # Row format:
        # state 0 - action 0
        # state 0 - action 1
        # ...
        # state 1 - action 0
        # ...
        # state |S-1| - action |A-1|
        self.N = lil_matrix((S * A, S))
        self.U = lil_matrix((S, 1))

        if logger_name:
            self.logger = logging.getLogger(logger_name)

    def __log(self, msg: str, level = logging.INFO):
        """Wrapper around logging."""
        if self.logger:
            self.logger.log(level, msg)

    def action_index(self, a: int) -> int:
        """Returns the index of a in the internal map.

        Args:
          a: action.

        Returns:
          index of a in the internal representations.
        """
        return a - 1

    def actions(self) -> List[int]:
        """Returns actions for this MDP."""

        return np.arange(1, self.action_space_size + 1)

    def get_utility(self, s: int) -> float:
        """Returns utility for state s."""

        s_index = self.state_index(s)
        return self.U[s_index][0]

    def set_reward(self, s: int, a: int, reward: float):
        """Sets reward for state s and action a."""

        s_index = self.state_index(s)
        a_index = self.action_index(a)
        self.rho[s_index, a_index] = reward

    def set_utility(self, s: int, utility: float):
        """Sets utility for state s."""

        s_index = self.state_index(s)
        self.U[s_index][0] = utility

    def state_index(self, s: int) -> int:
        """Returns the index of s in the internal map.

        Args:
          s: state.

        Returns:
          index of s in internal representations.
        """
        return s - 1

    def states(self) -> List[int]:
        """Returns list of states for the model."""

        return np.arange(1, self.state_space_size + 1)

    def row_index(self, s: int, a: int) -> int:
        """Returns the index of (s, a) in the internal counts map."""

        s_index = self.state_index(s)
        a_index = self.action_index(a)
        # states 2, actions 3
        # s0a0  0*3 + 0 = 0
        # s0a1  0*3 + 1 = 1
        # s0a2
        # s1a0  1*3 + 0 = 3
        return s_index*self.action_space_size + a_index

    def add_count(self, s: int, a: int, next_s: int):
        """Adds 1 to count matrix.

        Args:
          s: state
          a: action
          next_s: next state that results from state-action pair.
        """

        next_s_index = self.state_index(next_s)
        i = self.row_index(s, a)
        self.N[i, next_s_index] = self.N[i, next_s_index] + 1

    def lookahead(self, s: int, a: int) -> float:
        """Returns utility of performing action a from state s.

        Uses Bellman equation.

        Args:
          s: current state
          a: current action

        Returns:
          Utility given current state s and taking action a.
        """
        s_index = self.state_index(s)
        a_index = self.action_index(a)
        i = self.row_index(s, a)
        n = np.sum(self.N[i, :])
        self.__log(f"Lookahead, N({s}, {a}) = {n}")
        if n == 0:
            return 0.0

        r = self.rho[s_index, a_index] / n
        trans_prob = lambda next_state : self.N[i, self.state_index(next_state)] / n
        utilities = [trans_prob(s_next)*self.get_utility(s_next) for s_next in self.states()]
        utility = r + self.gamma * np.sum(utilities)
        self.__log(f"Lookahead, U({s}, {a}) = {utility}")
        return utility

    def backup(self, s: int) -> float:
        """Returns utility of optimal action from state s.

        Returns max of lookahead().

        Args:
          s: current state

        Returns:
          Utility of optimal action.
        """
        return np.max([self.lookahead(s, a) for a in self.actions()])

    def to_mdp(self) -> MDP:
        """Converts this instance to an equivalent MDP."""
        raise NotImplementedError("to_mdp() not implemented.")

    def update(self, s: int, a: int, r: float, next_s: int):
        """Updates model based on given parameters.

        Args:
          s: current state
          a: action to take from s
          r: reward for taking action a in state s
          next_s: next state to go to.
        """
        s_index = self.state_index(s)
        a_index = self.action_index(a)
        next_s_index = self.state_index(next_s)
        self.add_count(s, a, next_s)
        self.rho[s_index, a_index] = self.rho[s_index, a_index] + r
        self.planner.update(self, s_index, a_index, r, next_s_index)
