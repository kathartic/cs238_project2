from mdp import MaximumLikelihoodMDP

def e_greedy(model: MaximumLikelihoodMDP, epsilon: float, s: int) -> int:
    """Returns next action to take.

    Args:
      model: Model to choose actions from.
      epsilon: probability of choosing non-greedy action.
      s: state to take action from.

    Returns:
      action (in model representation) to take.
    """
    raise NotImplementedError("e_greedy() not implemented.")
