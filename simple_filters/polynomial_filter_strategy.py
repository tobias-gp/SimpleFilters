import numpy as np

from . import FilterStrategy

class PolynomialFilterStrategy(FilterStrategy): 
    """ 
    The filter strategy implements a simple Savitzky-Golay filter
    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

    A performance-oriented outlier detection is implemented via thresholding the median difference 
    to the median, multiplied by the outlier_rejection_ratio
    """

    def __init__(self, poly_degree=3, reject_outliers=True, outlier_rejection_ratio=2.0): 
        super().__init__()

        self.poly_degree = poly_degree
        self.reject_outliers = reject_outliers
        self.outlier_rejection_ratio = outlier_rejection_ratio
        self.history = None
        self.poly_fn = None

    def update(self, history): 
        self.history = history
        self.poly_fn = None

    def eval(self, time=0):
        # we center the time around the latest sample, which will be T=0
        history_size = self.history.shape[0]
        offset_time = history_size + time - 1 

        if self.history is None or history_size == 0: 
            return None

        # if the polynomial functions are not existent, calculate them
        if self.poly_fn is None: 
            self.set_polynomials()
            
        return self.eval_polynomials(offset_time)

    def eval_polynomials(self, t): 
        length = self.history.shape[1]
        predicted_states = np.zeros(length)

        for i in range(0, length): 
            predicted_states[i] = self.poly_fn[i](t)

        return predicted_states

    def set_polynomials(self): 
        self.poly_fn = []
        for i in range(0, self.history.shape[1]): 
            self.poly_fn.append(self.calc_polynomial(self.history[:, i]))

    def calc_polynomial(self, x): 
        length = x.shape[0]
        y = np.arange(length)

        if self.reject_outliers: 
            # reject outliers that are far awy from the median
            # determine the median of the mean absolute difference as a threshold
            delta = np.abs(x - np.median(x))
            rel_delta = delta / np.median(delta)
            
            mask = rel_delta < self.outlier_rejection_ratio
            x = x[mask]
            y = y[mask]

        # now fit the filter function
        coeffs = np.polyfit(y, x, self.poly_degree)
        poly_fn = np.poly1d(coeffs)

        return poly_fn
