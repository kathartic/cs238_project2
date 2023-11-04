import logging

from mdp import MaximumLikelihoodMDP
from typing import Hashable, List, Tuple

def simulate(
        model: MaximumLikelihoodMDP,
        policy,
        max_iter: int) -> List[Tuple[Hashable, Hashable]]:
    """Simulates using model, policy.

    Args:
      model: Model to simulate.
      policy: Policy to follow.
      max_iter: maximum iterations.

    Returns:
      trajectory taken in (state, action) pairs.
    """
    raise NotImplementedError("simulate() not implemented")


def write_policy(file_name: str,
                 S: List[Hashable],
                 A: List[Hashable],
                 trajectory: List[Tuple[Hashable, Hashable]]):
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
    logger.setLevel(logging.WARN)
    fh = logging.FileHandler(logger_name + '.log')
    fh.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
