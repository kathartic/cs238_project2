import logging
import pandas

from explore import EGreedy
from mdp import MaximumLikelihoodMDP


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


def set_counts(model: MaximumLikelihoodMDP,
               df: pandas.DataFrame):
    """Sets counts and rewards for the given model."""

    for _, row in df.iterrows():
        model.add_count(row['s'], row['a'], row['sp'])
        model.set_reward(row['s'], row['a'], row['r'])


def write_model_policy(file_name: str,
                 model: MaximumLikelihoodMDP,
                 logger: logging.Logger = None):
    """Writes policy to file <file_name>.policy.

    Each row i in the file corresponds to the action taken for the state i.

    Args:
      model: Model
    """
    egreedy = EGreedy(epsilon = 0)
    logger.info(f'Writing to file {file_name}.policy')
    with open(file_name + '.policy', 'w') as f:
        for state in model.states():
            best_action = egreedy(model, state)
            f.write("{}\n".format(best_action))


def make_logger(logger_name: str, level = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    fh = logging.FileHandler(logger_name + '.log')
    fh.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
