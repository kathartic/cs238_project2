import queue
import numpy as np

from planner import MDP, Planner
from typing import Hashable

class PrioritizedUpdate(Planner):
    """Implements prioritized update exploration."""

    def __init__(self, m: int):
        """Initializes the instance.
        
        Args:
          m: number of updates.          
        """

        self.m = m
        self.pq = queue.PriorityQueue()

    def __current_priority(self, s: Hashable) -> float:
        """Returns current priority for state s, or 0 if not present."""

        for (index, item) in enumerate(self.pq.queue()):
            priority = item.priority
            if item == s:
                del self.pq.queue[index]
                return priority
        return 0.0

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

        for s_bar in model.states():
            for a_bar in model.actions():
                i = model.row_index(s_bar, a_bar)
                n_sa = np.sum(model.N[i, :])
                if n_sa <= 0:
                    continue
                
                t = model.N[i, s_index] / n_sa
                priority = t * abs(model.get_utility(s) - u)
                if priority > 0:
                    current_priority = self.__current_priority(s_bar)
                    self.pq.put((max(priority, current_priority), s_bar))

    
    def update(self, model: MDP, s: Hashable, a: Hashable, r: float, next_s: Hashable):
        self.pq.put((np.inf, s))
        for i in range(self.m):
            if self.pq.qsize() == 0:
                break
            self.__update(model, self.pq.get(False)[1])


Planner.register(PrioritizedUpdate)
