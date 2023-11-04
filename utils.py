import logging
import numpy as np

from mdp import MaximumLikelihoodMDP
from typing import List, Tuple

def simulate(
        model: MaximumLikelihoodMDP,
        policy,
        max_iter: int) -> np.ndarray:
    """Simulates using model, policy.

    Args:
      model: Model to simulate.
      policy: Policy to follow.
      max_iter: maximum iterations.

    Returns:
      trajectory taken in (state, action) pairs.
    """

    s = np.random.choice(model.states())
    trajectory = np.zeros((max_iter, 2))
    for i in range(max_iter):
        a, next_state = model.simulate(policy, s)
        trajectory[i][0] = s
        trajectory[i][1] = a
        s = next_state
    return trajectory


def write_policy(file_name: str,
                 S: List[int],
                 A: List[int],
                 trajectory: List[Tuple[int, int]]):
    """Writes policy to file <file_name>.policy.

    Each row i in the file corresponds to the action taken for the state i.

    Args:
      file_name: filename to write to.
      S: state space.
      A: action space.
      trajectory: trajectory taken.
    """
    raise NotImplementedError("write_policy() not implemented.")


def make_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logger_name + '.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
