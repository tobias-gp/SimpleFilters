from unittest import TestCase
import numpy as np

from ..simple_filters import Filter, FilterStrategy, NumpyFilterStrategy

import pytest

class TestFilter(TestCase):

    def test_numpx_filter(self): 
        strategy = NumpyFilterStrategy(np.mean)
        filter = Filter(strategy, history_size=10)

        for i in range(0, 20): 
            filter.update([i, i + 1])

        result = filter.eval()

        self.assertEqual(result[0], 14.5)
        self.assertEqual(result[1], 15.5)
