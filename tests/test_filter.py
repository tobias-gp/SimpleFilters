from unittest import TestCase

from ..filter import Filter, FilterStrategy, MeanFilterStrategy

import pytest

class TestFilter(TestCase):

    def test_mean_filter(self): 
        strategy = MeanFilterStrategy()
        filter = Filter(strategy, history_size=10)

        for i in range(0, 20): 
            filter.update_state([i, i + 1])

        result = filter.get_filtered_state()

        self.assertEqual(result[0], 14.5)
        self.assertEqual(result[1], 15.5)
        
    def test_queue(self):
        strategy = MeanFilterStrategy()
        filter = Filter(strategy, history_size=10)

        for i in range(0, 20): 
            filter.update_state([i])

        state = filter.get_history()

        self.assertEqual(state.shape[0], 10)
        self.assertEqual(state[0], [10])
        self.assertEqual(state[9], [19])

        assert(True)
