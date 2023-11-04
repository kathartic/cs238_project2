import logging
import queue
import numpy as np

from planner import MDP, Planner
from typing import Hashable

class PrioritizedUpdate(Planner):
    """Implements prioritized update exploration."""

    def __init__(self, m: int, logger_name: str = None):
        """Initializes the instance.
        
        Args:
          m: number of updates. 
          loggerName: optional logging name.         
        """

        self.m = m
        # Entries are tuples of (priority: float, state: Hashable)
        self.pq = queue.PriorityQueue()
        if logger_name:
            self.logger = logging.getLogger(logger_name)

    def __current_priority(self, s: Hashable) -> float:
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

    def __update(self, model: MDP, s: Hashable):
        """Internal method for update.

        Called by public method. Given state s, updates its utility in the
        model AND updates its priority in the internal PQ.

        Args:
          s: state
        """

        u = model.get_utility(s)
        s_index = model.state_index(s)        
        model.set_utility(model.backup(s))
        new_utility = model.get_utility(s)
        self.__log(f"Updated utility for state {s}: {u} to {new_utility}")

        for s_bar in model.states():
            for a_bar in model.actions():
                i = model.row_index(s_bar, a_bar)
                n_sa = np.sum(model.N[i, :])
                if n_sa <= 0:
                    self.__log(f"No next state counts for action {a_bar}, state {s_bar}")
                    continue
                
                t = model.N[i, s_index] / n_sa
                priority = t * abs(new_utility - u)
                if priority > 0:
                    current_priority = self.__current_priority(s_bar)
                    updated_priority = max(priority, current_priority)
                    self.__log(f"Updating priority of state {s_bar} to {updated_priority}")
                    self.pq.put((updated_priority, s_bar))

    
    def update(self, model: MDP, s: Hashable, a: Hashable, r: float, next_s: Hashable):
        self.pq.put((np.inf, s))
        for i in range(self.m):
            if self.pq.qsize() == 0:
                self.__log("No items left in queue, exiting early.", logging.WARN)
                break
            self.__update(model, self.pq.get(False)[1])


Planner.register(PrioritizedUpdate)
