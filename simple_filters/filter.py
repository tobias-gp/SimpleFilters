from collections import deque
import numpy as np

class FilterStrategy: 

    def __init__(self): 
        self.filter = None

    def update(self, history): 
        raise NotImplementedError("Abstract base function called")

    def eval(self, time=0): 
        raise NotImplementedError("Abstract base function called")

class Filter: 
    """
    Implements a filter with a LIFO queue, according to the history size specified. 
    A filter strategy must be supplied, which does the actual filtering
    """

    def __init__(self, strategy, history_size=10): 
        self.history = None
        self.history_size = history_size
        self.strategy = strategy

    def update(self, state):
        if self.history is None: 
            self.history = np.array([state])
            self.strategy.update(self.history)
            return 

        # a ring buffer would of course be more efficient 
        # but require resorting when applying the filter strategies
        history_length = self.history.shape[0]
        if history_length >= self.history_size: 
            self.history = self.history[1:]
        
        self.history = np.append(self.history, [state], axis=0)
        self.strategy.update(self.history)

    def get_history(self): 
        return self.history

    def eval(self, time=0): 
        return self.strategy.eval(time)

    def raw(self, time=0): 
        # we center the time around the latest sample, which will be T=0
        history_size = self.history.shape[0]
        offset_time = history_size + time - 1 

        return self.history[offset_time]