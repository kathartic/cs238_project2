import abc

class MDP(abc.ABC):
    """Abstract class defining an MDP."""

    @abc.abstractmethod
    def lookahead(self, s, a) -> float:
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
    def backup(self, s) -> float:
        """Returns utility of optimal action from state s.
        
        Returns max of lookahead().

        Args:
          s: current state

        Returns:
          Utility of optimal action.
        """
        pass

    @abc.abstractmethod
    def update(self, s, a, r, next_s):
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
    def update(self, model: MDP, s: int, a: int, r: float, next_s: int):
        """Mutates given model."""