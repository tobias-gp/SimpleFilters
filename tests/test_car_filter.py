from unittest import TestCase

from ..filter import Filter, CarFilterStrategy

import pytest

class TestCarFilter(TestCase):

    def setUp(self): 
        self.history_size = 10


    def test_single(self): 
        strategy = CarFilterStrategy(update_rate_in_ms=100, poly_order=3)
        self.filter = Filter(strategy, history_size=self.history_size)
        self.generate_linear_state_history()

        result = self.filter.get_filtered_state()

        self.assertEqual(result.shape[0], 1)
        self.assertAlmostEqual(result[0,0], 9.)
        self.assertAlmostEqual(result[0,1], 10.)
        self.assertAlmostEqual(result[0,2], 11.)

    def test_future(self): 
        strategy = CarFilterStrategy(update_rate_in_ms=100, poly_order=3, predict_samples=2)
        self.filter = Filter(strategy, history_size=self.history_size)
        self.generate_linear_state_history()

        result = self.filter.get_filtered_state()

        self.assertEqual(result.shape[0], 2)
        
        self.assertAlmostEqual(result[0,0], 9.)
        self.assertAlmostEqual(result[0,1], 10.)
        self.assertAlmostEqual(result[0,2], 11.)

        self.assertAlmostEqual(result[1,0], 10.)
        self.assertAlmostEqual(result[1,1], 11.)
        self.assertAlmostEqual(result[1,2], 12.)


    def generate_linear_state_history(self): 
        """
        Generates a simple linear succession of values 
        """
        for i in range(0, self.history_size): 
            self.filter.update_state([i, i + 1, i + 2, i + 3])