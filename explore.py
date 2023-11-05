import logging
import numpy as np

from mdp import MaximumLikelihoodMDP


class EGreedy():
    """Implements e-greedy policy."""

    def __init__(self,
                 epsilon: float = 0.3,
                 alpha: float = 0.97,
                 logger: logging.Logger = None):

        self.epsilon = epsilon
        self.alpha = alpha
        self.logger = logger

    def __call__(self, model: MaximumLikelihoodMDP, s: int) -> int:
      """Returns next action to take.

      Args:
        model: Model to choose actions from.
        epsilon: probability of choosing non-greedy action.
        s: state to take action from.

      Returns:
        action (in model representation) to take.
      """
      prev_epsilon = self.epsilon
      self.epsilon = prev_epsilon * self.alpha
      if np.random.random() < prev_epsilon:
          new_action = np.random.choice(model.actions())
          if self.logger:
              self.logger.info(f'Egreedy: randomly chose action {new_action}')

      max_a = 1
      q_max = -np.inf
      for action in model.actions():
          q_sa = model.lookahead(s, action)
          if q_sa > q_max:
              q_max = q_sa
              max_a = action
      if self.logger:
          self.logger.info(f'Egreedy: chose action {max_a} with Q({s}, {max_a}) = {q_max}')
      return max_a
