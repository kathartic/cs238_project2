import logging
import pandas
import numpy as np
import time
import sys
import utils

from explore import e_greedy
from mdp import MaximumLikelihoodMDP
from prioritizedupdate import PrioritizedUpdate
from typing import Tuple


def get_gamma(file_name: str) -> float:
    """Returns discount factor for supported filenames.

    Values taken from course website.

    Args:
      file_name: supported filenames are "small", "medium", or "large".

    Returns:
      discount factor
    """
    if file_name == "small":
        return 0.95
    elif file_name == "medium":
        return 1.0
    elif file_name == "large":
        return 0.95
    else:
        raise ValueError(f"Unsupported file: {file_name}")


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


def set_counts(model: MaximumLikelihoodMDP,
               df: pandas.DataFrame,
               logger: logging.Logger):
    """Sets counts and rewards for the given model."""

    for _, row in df.iterrows():
        model.add_count(row['s'], row['a'], row['sp'])
        model.set_reward(row['s'], row['a'], row['r'])
    logger.info(f'Counts: {model.N.toarray()}')
    logger.info(f'Rewards: {model.rho.toarray()}')


def main():
    if len(sys.argv) != 2:
        raise Exception("usage: python project2.py <infile>.csv")

    file_name = sys.argv[1]
    logger = utils.make_logger(file_name)
    start = time.time()

    # Set up MDP, planner.
    gamma = get_gamma(file_name)
    df, S, A = read_data(file_name, logger)
    update_count = 2  # TODO(kathuan): tune this
    max_iter = S  # TODO(kathuan): tune this
    planner = PrioritizedUpdate(update_count, file_name)
    model = MaximumLikelihoodMDP(S, A, gamma, planner, file_name)
    set_counts(model, df, logger)

    # Run simulation and write output.
    trajectory = utils.simulate(model, e_greedy, max_iter)
    end = time.time()
    logger.critical(f"Elapsed time in seconds: {end - start}")
    utils.write_policy(file_name, S, A, trajectory)


if __name__ == '__main__':
    main()