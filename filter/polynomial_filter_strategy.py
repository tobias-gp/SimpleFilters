import numpy as np

from . import FilterStrategy

class PolynomialFilterStrategy(FilterStrategy): 
    """ 
    The filter strategy implements a simple Savitzky-Golay filter
    https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

    A performance-oriented outlier detection is implemented via thresholding the median difference 
    to the median, multiplied by the outlier_rejection_ratio
    """

    def __init__(self, poly_degree=3, predict_samples=1, reject_outliers=True, outlier_rejection_ratio=2.0): 
        self.poly_degree = poly_degree
        self.predict_samples = predict_samples
        self.reject_outliers = reject_outliers
        self.outlier_rejection_ratio = outlier_rejection_ratio

    def apply(self, history):
        if history is None or history.shape[0] == 0: 
            return None

        predicted_states = np.zeros((self.predict_samples, history.shape[1]))

        for i in range(0,4): 
            predicted_states[:, i] = self.apply_to_series(history[:, i])

        return predicted_states

    def apply_to_series(self, x): 
        length = x.shape[0]
        y = np.arange(length)
        predicted = np.zeros(self.predict_samples)

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

        # now evaluate the functions at time points t E (0, ..., predict_samples)
        for s in range(0, self.predict_samples): 
            t = length - 1 + s
            predicted[s] =  poly_fn(t)

        return predicted
