import logging
import numpy as np

from scipy.sparse import lil_matrix
from typing import List, Tuple


class MDP:
    """Defines a MDP."""

    def __init__(self,
                 S: int,
                 A: int,
                 T,
                 R):
        """Instantiates an instance.

        Args:
          S: Size of state space. Assumes states are 1 to S, inclusive.
          A: Size of action space. Assumes states are 1 to S, inclusive.
          T: Transition matrix.
          R: Reward matrix (i = state, j = action).
        """
        self.S = S
        self.A = A
        self.T = T
        self.R = R

    def action_index(self, a: int) -> int:
        """Returns the index of a in the internal map.

        Args:
          a: action.

        Returns:
          the representation of a as an int that can be indexed in the internal
          count matrix.
        """
        return a - 1

    def state_index(self, s: int) -> int:
        """Returns the index of s in the internal map.

        Args:
          s: state.

        Returns:
          the representation of s as an int that can be indexed in the internal
          count matrix.
        """
        return s - 1

    def row_index(self, s: int, a: int) -> int:
        """Returns the index of (s, a) in the internal counts map."""

        s_index = self.state_index(s)
        a_index = self.action_index(a)
        # states 3, actions 2
        # a0s0  0*3 + 0 = 0
        # a0s1  0*3 + 1 = 1
        # a0s2  0*3 + 2 = 2
        # a1s0  1*3 + 0 = 3
        # a1s1  1*3 + 1 = 4
        return a_index*self.S + s_index

    def transition_prob(self, s: int, a: int, next_s: int) -> float:
        i = self.row_index(s, a)
        return self.T[i, next_s]

    def TR(self, s: int, a: int) -> Tuple[int, float]:
        """Samples transition and reward.

        Args:
          s: state to take action from.
          a: action to take.

        Returns:
          Tuple of next state, reward.
        """
        s_index = self.state_index(s)
        a_index = self.action_index(a)
        i = self.row_index(s, a)
        probs = np.array(self.T[i, :].toarray()).reshape((self.S,))
        next_state = np.random.choice(np.arange(1, self.S + 1),
                                      p = probs)
        return (next_state, self.R[s_index, a_index])


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
        self.S = S
        self.A = A
        self.rho = lil_matrix((S, A))
        self.planner = planner
        # Maps action-state pairs, to potential next states.
        # Row format:
        # action 0 - state 0
        # action 0 - state 1
        # ...
        # action 0 - state S-1
        # action 1 - action 0
        # ...
        # action A-1 - state S-1
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

        return np.arange(1, self.A + 1)

    def get_utility(self, s: int) -> float:
        """Returns utility for state s."""

        s_index = self.state_index(s)
        return self.U[s_index, 0]

    def set_reward(self, s: int, a: int, reward: float):
        """Sets reward for state s and action a."""

        s_index = self.state_index(s)
        a_index = self.action_index(a)
        self.rho[s_index, a_index] = reward

    def set_utility(self, s: int, utility: float):
        """Sets utility for state s."""

        s_index = self.state_index(s)
        self.U[s_index, 0] = utility

    def state_index(self, s: int) -> int:
        """Returns the index of s in the internal map.

        Args:
          s: state.

        Returns:
          index of s in internal representations.
        """
        return s - 1

    def states(self) -> List[int]:
        """Returns sorted list of states for the model."""

        return np.arange(1, self.S + 1)

    def row_index(self, s: int, a: int) -> int:
        """Returns the index of (s, a) in the internal counts map."""

        s_index = self.state_index(s)
        a_index = self.action_index(a)
        # states 3, actions 2
        # a0s0  0*3 + 0 = 0
        # a0s1  0*3 + 1 = 1
        # a0s2  0*3 + 2 = 2
        # a1s0  1*3 + 0 = 3
        # a1s1  1*3 + 1 = 4
        return a_index*self.S + s_index

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
        if n == 0:
            return 0.0

        r = self.rho[s_index, a_index] / n
        trans_prob = lambda next_state : self.N[i, self.state_index(next_state)] / n
        utilities = [trans_prob(s_next)*self.get_utility(s_next) for s_next in self.states()]
        utility = r + self.gamma * np.sum(utilities)
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

        # N(s, a, s') += 1, rho[s, a] += r
        self.add_count(s, a, next_s)
        self.rho[s_index, a_index] = self.rho[s_index, a_index] + r
        self.planner.update(self, s_index, a_index, r, next_s_index)

    def to_mdp(self) -> MDP:
        """Converts this instance to an equivalent MDP."""

        # N_sa is a matrix holding N(s,a). Each row is a state, each col is an
        # action.
        N_sa = self.N.sum(1)
        N_sa = N_sa.reshape((self.A, self.S))
        N_sa = np.array(N_sa.transpose())  # |S| x |A|

        # R(s,a) = rho[s,a] / N(s,a) if N(s, a) > 0 else 0
        R = lil_matrix(np.divide(self.rho.toarray(), N_sa, out=np.zeros(self.rho.shape), where=(N_sa != 0)))

        # T(s,a) = N(s, a, s') / N(s, a) if N(s, a) > 0 else 0
        T = lil_matrix(self.N.shape)
        for a in self.actions():
            # Get counts for all (state, action) pairs with action == a.
            action_index = self.action_index(a)
            divisor = N_sa[:, action_index]  # N(s, a == a), |S| x 1 matrix

            # Get all N(s, a = a, s').
            start_index = self.row_index(1, a)
            end_index = self.row_index(self.S, a) + 1
            counts = self.N[start_index:end_index, :]

            # Perform division along the row, so each entry of divisor divides
            # a row in counts. All the transposing is empirically chosen.
            T_sa = np.divide(
                counts.toarray().T,
                divisor,
                out=np.zeros((self.S, self.S)),
                where=(divisor != 0))
            T[start_index:end_index, :] = T_sa.T

        return MDP(self.S, self.A, T, R)

    def simulate(self, policy, s: int) -> Tuple[int, int]:
        """Simulates one round using policy.

        Args:
          policy: callable policy to follow.
          s: start state.

        Returns:
          tuple of (action taken, next state)
        """
        mdp = self.to_mdp()
        a = policy(self, s)
        next_s, r = mdp.TR(s, a)
        self.__log(f"Update after simulation: (s, a, r, s'): ({s}, {a}, {r}, {next_s})")
        self.update(s, a, r, next_s)
        return (a, next_s)
