import time
from unittest import TestCase
import pytest

from ..simple_filters import Filter, PolynomialFilterStrategy

class TestCarFilter(TestCase):

    def setUp(self): 
        self.history_size = 10

    def test_single(self): 
        strategy = PolynomialFilterStrategy(poly_degree=3)
        self.filter = Filter(strategy, history_size=self.history_size)
        self.generate_linear_state_history()

        result = self.filter.eval()

        self.assertEqual(result.shape[0], 4)

        self.assertAlmostEqual(result[0], 9.)
        self.assertAlmostEqual(result[1], 10.)
        self.assertAlmostEqual(result[2], 11.)
        self.assertAlmostEqual(result[3], 12.)

    def test_future(self): 
        strategy = PolynomialFilterStrategy(poly_degree=3)
        self.filter = Filter(strategy, history_size=self.history_size)
        self.generate_linear_state_history()

        # current result at t=0
        result = self.filter.eval()
        self.assertAlmostEqual(result[0], 9.)
        self.assertAlmostEqual(result[1], 10.)
        self.assertAlmostEqual(result[2], 11.)
        self.assertAlmostEqual(result[3], 12.)

        # predict the future
        result = self.filter.eval(time=1)
        self.assertAlmostEqual(result[0], 10.)
        self.assertAlmostEqual(result[1], 11.)
        self.assertAlmostEqual(result[2], 12.)
        self.assertAlmostEqual(result[3], 13.)

    def test_execution_time(self): 
        strategy = PolynomialFilterStrategy(poly_degree=3)
        self.filter = Filter(strategy, history_size=self.history_size)
        self.generate_linear_state_history()

        start_time = time.time()

        for i in range(0, 1000): 
            self.filter.eval()

        delta_in_ms = (time.time() - start_time)

        print("Execution took %f ms per filtering operation" %  delta_in_ms)
        self.assertLessEqual(delta_in_ms, 2.0)

    def test_outliers(self): 
        strategy = PolynomialFilterStrategy(poly_degree=3, outlier_rejection_ratio=2.0)
        self.filter = Filter(strategy, history_size=self.history_size)
        self.generate_outlier_state_history()

        result = self.filter.eval()
        self.assertAlmostEqual(result[0], 9.)
        self.assertAlmostEqual(result[1], 10.)
        self.assertAlmostEqual(result[2], 11.)
        self.assertAlmostEqual(result[3], 12.)

        result = self.filter.eval(time=1)
        self.assertAlmostEqual(result[0], 10.)
        self.assertAlmostEqual(result[1], 11.)
        self.assertAlmostEqual(result[2], 12.)
        self.assertAlmostEqual(result[3], 13.)

    def generate_linear_state_history(self): 
        """
        Generates a simple linear succession of values 
        """
        for i in range(0, self.history_size): 
            self.filter.update([i, i + 1, i + 2, i + 3])

    def generate_outlier_state_history(self): 
        """
        Generates a simple linear succession of values 
        """
        
        outlier_frequency = 3
        outlier_offset = 20 # generating a clear outlier

        for i in range(0, self.history_size): 
            if i % outlier_frequency == 0: 
                self.filter.update([i + outlier_offset, i + 1 + outlier_offset, i + 2 + outlier_offset, i + 3 + outlier_offset])
            else: 
                self.filter.update([i, i + 1, i + 2, i + 3])

        