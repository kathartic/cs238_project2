import logging
import mdp
import numpy as np
import pandas

from typing import List, Set, Tuple

class QLearning(mdp.IndexAdapter):
    def __init__(self,
                 S: int,
                 A: int,
                 gamma: float,
                 Q,
                 alpha: float,
                 df: pandas.DataFrame,
                 known_states: Set[int] = None,
                 logger: logging.Logger = None):
        self.S = S
        self.A = A
        self.gamma = gamma
        self.Q = Q
        self.logger = logger
        self.alpha = alpha
        self.df = df
        self.known_states = known_states

    def states(self) -> List[int]:
        return np.arange(1, self.S + 1)

    def actions(self) -> List[int]:
        return np.arange(1, self.A + 1)

    def __choices(self) -> List[int]:
        return list(self.known_states) if self.known_states is not None else self.states()

    def next_state(self, s: int, a: int) -> Tuple[int, float]:
        """Returns tuple of next state, reward."""

        queried_df = self.df.query('s == @s & a == @a')
        if queried_df.size == 0:
            if self.logger is not None:
                self.logger.warning(f'No results found for (s = {s}, a = {a}).')
            return (np.random.choice(self.__choices()), -225.0)
        total_rows = queried_df.shape[0]
        random_row = np.random.choice(total_rows)
        row = queried_df.iloc[random_row]
        next_s = row.loc['sp']
        r = row.loc['r']
        if self.logger is not None:
            self.logger.info(f'Next action: {next_s}, r: {r}')
        return (next_s, r)

    def lookahead(self, s:int, a:int):
        return self.Q[self.state_index(s), self.action_index(a)]

    def update(self, s: int, a: int, r: float, next_s: int):
        s_index = self.state_index(s)
        a_index = self.action_index(a)
        next_s_index = self.state_index(next_s)
        discounted_max = self.gamma*np.max(self.Q[next_s_index, :].toarray())
        q_sa = self.Q[s_index, a_index]
        self.Q[s_index, a_index] = q_sa + self.alpha*(r + discounted_max - q_sa)

    def simulate(
            self,
            policy,
            max_iter: int):
        choices = self.__choices()
        s = np.random.choice(choices)
        loop_count = 0
        for _ in range(max_iter):
            a = policy(self, s)
            next_s, r = self.next_state(s, a)
            self.update(s, a, r, next_s)

            # Loop detection.
            if next_s == s:
                loop_count += 1
            else:
                loop_count = 0
            if loop_count > 10:
                next_s = np.random.choice(choices)
                if self.logger is not None:
                    self.logger.warning(f'Detected loop, going to random state {next_s}')

            s = next_s


def write_policy(file_name: str, model: QLearning, logger: logging.Logger = None):
    with open(file_name + '.policy', 'w') as f:
        for state in model.states():
            # Add one for action offset.
            Q_s = model.Q[model.state_index(state), :].toarray().reshape(-1)
            best_action = np.argmax(Q_s) + 1
            if Q_s.any() != 0:
                logger.info(f"Q[{state}, :] = {Q_s}; best value is {best_action}")
            f.write("{}\n".format(best_action))
