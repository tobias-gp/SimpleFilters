import numpy as np

from . import FilterStrategy

class MeanFilterStrategy(FilterStrategy): 

    def __init__(self): 
        pass

    def apply(self, history): 
        if history is None or history.shape[0] == 0: 
            return None

        return np.mean(history, axis=0)