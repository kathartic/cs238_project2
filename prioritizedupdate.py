import logging
import queue
import numpy as np

from mdp import MaximumLikelihoodMDP
from planner import Planner

class PrioritizedUpdate(Planner):
    """Implements prioritized update exploration."""

    def __init__(self, m: int, logger_name: str = None):
        """Initializes the instance.

        Args:
          m: number of updates.
          loggerName: optional logging name.
        """

        self.m = m
        # Entries are tuples of (priority: float, state: int)
        self.pq = queue.PriorityQueue()
        if logger_name:
            self.logger = logging.getLogger(logger_name)

    def __current_priority(self, s: int) -> float:
        """Returns current priority for state s, or 0 if not present."""

        for (index, item) in enumerate(self.pq.queue()):
            priority = item[0]
            if item == s:  # Remove from queue and return its priority.
                del self.pq.queue[index]
                return priority

        self.__log(f"State {s} not found in priority queue", logging.WARN)
        return 0.0

    def __log(self, msg: str, level = logging.INFO):
        """Wrapper around logging."""
        if self.logger:
            self.logger.log(level, msg)

    def __update(self, model: MaximumLikelihoodMDP, s: int):
        """Internal method for update.

        Called by public method. Given state s, updates its utility in the
        model AND updates its priority in the internal PQ.

        Args:
          s: state
        """

        u = model.get_utility(s)
        model.set_utility(model.backup(s))
        new_utility = model.get_utility(s)
        self.__log(f"Updated utility for state {s}: {u} to {new_utility}")
        utility_diff = abs(new_utility - u)
        if utility_diff == 0:
            return

        mdp = model.to_mdp()
        for s_bar in model.states():
            for a_bar in model.actions():
                t = mdp.transition_prob(s, a_bar, s_bar)
                if t == 0:
                    continue

                current_priority = self.__current_priority(s_bar)
                updated_priority = max(t * abs(new_utility - u), current_priority)
                self.__log(f"Updating priority of state {s_bar} to {updated_priority}")
                self.pq.put((updated_priority, s_bar))


    def update(self, model: MaximumLikelihoodMDP, s: int, a: int, r: float, next_s: int):
        self.pq.put((np.inf, s))
        for i in range(self.m):
            if self.pq.qsize() == 0:
                self.__log("No items left in queue, exiting early.", logging.WARN)
                break
            self.__update(model, self.pq.get(False)[1])


Planner.register(PrioritizedUpdate)
