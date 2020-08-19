from unittest import TestCase

from ..simple_filters import Tracker, TrackedObject, Filter, PolynomialFilterStrategy

import pytest
import numpy as np

class TestSingleStepSingleObjectTracking(TestCase):

    def setUp(self):
        strategy = PolynomialFilterStrategy()
        filter_prototype = Filter(strategy, history_size=1)
        self.tracker = Tracker(filter_prototype, distance_threshold=1.)
    
    def test_main(self):
        states = [
            np.array([1.0, 1.0]),
            np.array([1.5, 1.5]),
            np.array([2.0, 2.0]),
            np.array([3.0, 3.0]),
            np.array([3.5, 3.5]),
            np.array([4.0, 4.0])
        ]
        
        expected_tracking_ids = [1, 1, 1, 2, 2, 2]

        for i, (state, expected_tracking_id) in enumerate(zip(states, expected_tracking_ids)):
            print("timestep", i)
            self.tracker.update(state)
            tracked_state = self.tracker.get_tracked_objects()

            self.assertEqual(1, len(tracked_state))
            self.assertEqual(tracked_state[0].id, expected_tracking_id)