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

    def __init__(self, strategy, history_size=10, update_rate_in_ms=100): 
        self.history = None
        self.history_size = history_size
        self.strategy = strategy

    def update(self, state):
        if self.history is None: 
            self.history = np.array([state])
            return 

        # a ring buffer would of course be more efficient 
        # but require resorting when interpolating
        history_length = self.history.shape[0]
        if history_length >= self.history_size: 
            self.history = self.history[1:]
        
        self.history = np.append(self.history, [state], axis=0)
        self.strategy.update(self.history)

    def get_history(self): 
        return self.history

    def eval(self, time=0): 
        return self.strategy.eval(time)
