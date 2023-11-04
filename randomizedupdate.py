import logging
import numpy as np

from mdp import MaximumLikelihoodMDP
from planner import Planner


class RandomizedUpdate(Planner):
    """Implements randomized updates."""

    def __init__(self, m: int, logger_name: str = None):
        """Initializes the instance.

        Args:
          m: number of updates.
        """
        self.m = m
        if logger_name:
            self.logger = logging.getLogger(logger_name)


    def __log(self, msg: str, level = logging.INFO):
        """Wrapper around logging."""
        if self.logger:
            self.logger.log(level, msg)


    def update(self, model: MaximumLikelihoodMDP, s: int, a: int, r: int, next_s: int):
        new_states = np.random.choice(model.states(), self.m, replace=False)
        new_states = np.concatenate([s], new_states, axis=None)
        assert new_states[0] == s, "Needs to work this way"
        for state in new_states:
            u = model.get_utility(state)
            model.set_utility(state, model.backup(state))
            new_utility = model.get_utility(state)
            self.__log(f"Updated utility for state {state}: {u} to {new_utility}")