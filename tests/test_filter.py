from unittest import TestCase

from ..simple_filters import Filter, FilterStrategy, DummyFilterStrategy

import pytest

class TestFilter(TestCase):

    def test_dummy_filter(self): 
        strategy = DummyFilterStrategy()
        filter = Filter(strategy, history_size=10)

        for i in range(0, 20): 
            filter.update([i, i + 1])

        result = filter.eval()

        self.assertEqual(result[0], 19)
        self.assertEqual(result[1], 20)
        
    def test_queue(self):
        strategy = DummyFilterStrategy()
        filter = Filter(strategy, history_size=10)

        for i in range(0, 20): 
            filter.update([i])

        state = filter.get_history()

        self.assertEqual(state.shape[0], 10)
        self.assertEqual(state[0], [10])
        self.assertEqual(state[9], [19])

        assert(True)
