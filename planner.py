import abc

from mdp import MaximumLikelihoodMDP
from typing import Hashable

class Planner(abc.ABC):
    """Abstract class defining a planner."""
    @abc.abstractmethod
    def update(self, model: MaximumLikelihoodMDP, s: Hashable, a: Hashable, r: float, next_s: Hashable):
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
        pass
