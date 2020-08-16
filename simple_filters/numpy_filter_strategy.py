import numpy as np

from . import FilterStrategy

class NumpyFilterStrategy(FilterStrategy): 
    """
    Simply returns the latest item
    """

    def __init__(self, numpy_function): 
        super().__init__()
        self.history = None
        self.__numpy_function = numpy_function

    def update(self, history): 
        self.history = history

    def eval(self, time=0): 
        if self.history is None or self.history.shape[0] == 0: 
            return None

        return self.__numpy_function(self.history, axis=0)