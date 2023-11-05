import logging
import pandas
import numpy as np
import time
import utils

from explore import EGreedy
from mdp import MaximumLikelihoodMDP
from randomizedupdate import RandomizedUpdate
from typing import Tuple


def read_data(file_name: str,
              logger: logging.Logger) -> Tuple[pandas.DataFrame, int, int]:
    """Opens file and reads data.

    Args:
      file_name: name of file to read.
      logger: logging instance.

    Returns:
      tuple of df, state space, action space.
    """

    df = pandas.read_csv(file_name + ".csv", dtype=np.int32)
    S = df.loc[:, 's'].unique()
    S.sort()
    A = df.loc[:, 'a'].unique()
    A.sort()

    logger.info(f'State space ({S.shape}): {S}')
    logger.info(f'Action space ({A.shape}): {A}')
    return (df, S.size, A.size)


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
        a, next_s = model.simulate(policy, s)
        trajectory[i][0] = s
        trajectory[i][1] = a
        s = next_s
    return trajectory


def main():
    file_name = "small"
    logger = utils.make_logger(file_name)
    start = time.time()

    # Set up MDP, planner.
    gamma = utils.get_gamma(file_name)
    df, S, A = read_data(file_name, logger)
    update_count = 8  # TODO(kathuan): tune this
    max_iter = 300  # TODO(kathuan): tune this
    planner = RandomizedUpdate(update_count, file_name)
    model = MaximumLikelihoodMDP(S, A, gamma, planner, file_name)
    utils.set_counts(model, df)

    # Run simulation and write output.
    _ = simulate(model, EGreedy(), max_iter)
    end = time.time()
    logger.critical(f"Elapsed time in seconds: {end - start}")
    utils.write_model_policy(file_name, model)


if __name__ == '__main__':
    main()