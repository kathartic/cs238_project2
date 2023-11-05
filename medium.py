import logging
import pandas
import numpy as np
import qlearning
import time
import utils

from explore import EGreedy
from mdp import MaximumLikelihoodMDP
from randomizedupdate import RandomizedUpdate
from scipy.sparse import lil_matrix
from typing import Tuple


def read_data(file_name: str,
              logger: logging.Logger) -> Tuple[pandas.DataFrame, np.ndarray, int]:
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
    return (df, S, A.size)


def main():
    file_name = "medium"
    logger = utils.make_logger(file_name)
    start = time.time()

    # Set up MDP, planner.
    gamma = utils.get_gamma(file_name)
    df, S, A = read_data(file_name, logger)
    state_size = 50000
    max_iter = 70000  # TODO(kathuan): tune this
    Q = lil_matrix((state_size, A))
    qlearn_model = qlearning.QLearning(state_size, A, gamma, Q, 0.2, df, logger)

    # Run simulation and write output.
    try:
      qlearning.simulate_maximum_likelihood(
          qlearn_model, EGreedy(epsilon=0.3, logger=logger), max_iter, set(S))
    except Exception as e:
      end = time.time()
      logger.critical("Unrecoverable failure")
      logger.critical(f"Elapsed time in seconds: {end - start}")
      qlearning.write_policy(file_name, qlearn_model, logger)
      raise(e)

    end = time.time()
    logger.critical(f"Elapsed time in seconds: {end - start}")
    qlearning.write_policy(file_name, qlearn_model, logger)


if __name__ == '__main__':
    main()