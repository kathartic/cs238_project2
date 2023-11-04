import time
import sys
import utils

from explore import e_greedy
from prioritizedupdate import PrioritizedUpdate
from rmax import Rmax
from typing import Hashable, List, Tuple


def read_data(file_name: str) -> Tuple[List[Hashable], List[Hashable], float, float]:
    """Opens file and reads data.

    Returns:
      tuple of state space, action space, rmax, and gamma.
    """
    raise NotImplementedError("read_data() not implemented.")


def main():
    if len(sys.argv) != 2:
        raise Exception("usage: python project1.py <infile>.csv")

    file_name = sys.argv[1]

    # Set up logging.
    logger_name = "project1_small"
    logger = utils.make_logger(logger_name)

    start = time.time()

    # Set up MDP, planner.
    S, A, rmax = read_data(file_name)
    exploration_threshold = 3  # TODO(kathuan): tune this
    update_count = 2  # TODO(kathuan): tune this
    max_iter = 100
    planner = PrioritizedUpdate(update_count, logger_name)
    model = Rmax(S, A, 0.95, planner, exploration_threshold, rmax, logger_name)

    # Run simulation and write output.
    trajectory = utils.simulate(model, e_greedy, max_iter)
    end = time.time()
    logger.critical(f"Elapsed time in seconds: {end - start}")
    utils.write_policy(file_name, S, A, trajectory)


if __name__ == '__main__':
    main()