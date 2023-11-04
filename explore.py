import numpy as np

from mdp import MaximumLikelihoodMDP

# sure, why  not. have some globals.
alpha = 0.99
epsilon = 0.2

def e_greedy(model: MaximumLikelihoodMDP, s: int) -> int:
    """Returns next action to take.

    Args:
      model: Model to choose actions from.
      epsilon: probability of choosing non-greedy action.
      s: state to take action from.

    Returns:
      action (in model representation) to take.
    """
    prev_epsilon = epsilon
    epsilon = epsilon * alpha
    if np.random.random() < prev_epsilon:
        return np.random.choice(model.actions())

    max_a = 1
    q_max = -np.inf
    for action in model.actions():
        q_sa = model.lookahead(s, action)
        if q_sa > q_max:
            q_max = q_sa
            max_a = action
    return max_a
