import logging

from prioritizedupdate import PrioritizedUpdate
from rmax import Rmax

def simulate():
    """Simulates."""
    pass


def read_data():
    """Opens file and reads data."""
    pass


def make_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARN)
    fh = logging.FileHandler('small.log')
    fh.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def main():
    # Set up logging.
    logger_name = "project1_small"
    logger = make_logger(logger_name)

    # Set up MDP, planner.
    S, A, rmax = read_data()
    exploration_threshold = 3  # TODO(kathuan): tune this
    update_count = 2  # TODO(kathuan): tune this
    planner = PrioritizedUpdate(update_count, logger_name)
    model = Rmax(S, A, 0.95, planner, exploration_threshold, rmax, logger_name)


if __name__ == '__main__':
    main()