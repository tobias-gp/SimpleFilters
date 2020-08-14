import numpy as np

from ..filter import FilterStrategy



class CarFilterStrategy(FilterStrategy): 
    """ 
    A car's state is described by the following state: 
    * tx: 0
    * ty: 1
    * tz: 2
    * rotation: 3

    The filter strategy implements a simple Savitzky-Golay filter for tranlsation X/Y/Z
    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

    """

    number_of_dimensions = 4

    def __init__(self, update_rate_in_ms=500, poly_order=3, predict_samples=1): 
        self.update_rate_in_ms = update_rate_in_ms
        self.poly_order = poly_order
        self.predict_samples = predict_samples

    def apply(self, history):
        """
        Filter the position of the car by interpolating
        """

        history_size = history.shape[0]
        predicted_states = np.zeros((self.predict_samples, CarFilterStrategy.number_of_dimensions))
        poly_fn = []

        if history_size < 2: 
            return history[-1]

        # iterate over the translation indices 
        # estimate the polynomial functions for each index
        y = np.arange(history_size)

        for i in range(0,3): 
            x = history[:,i]
            coeffs = np.polyfit(y, x, self.poly_order)
            poly_fn.append(np.poly1d(coeffs))

        # now evaluate the functions at time points t E (0, ..., predict_samples)
        for i in range(0, 3): 
            for s in range(0, self.predict_samples): 
                t = history_size - 1 + s
                predicted_states[s, i] =  poly_fn[i](t)

                # TODO: Implement majority voting
                # predicted_states[s, 4] = 

        return predicted_states

