import numpy as np

from . import FilterStrategy

class DummyFilterStrategy(FilterStrategy): 
    """
    Simply returns the latest item
    """

    def __init__(self): 
        super().__init__()
        self.history = None

    def update(self, history): 
        self.history = history

    def eval(self, time=0): 
        if self.history is None or self.history.shape[0] == 0: 
            return None

        return self.history[self.history.shape[0] - 1]