from unittest import TestCase

from ..simple_filters import Tracker, TrackedObject, Filter, PolynomialFilterStrategy, DummyFilterStrategy

import pytest
import numpy as np

class TestSingleStepSingleObjectTracking(TestCase):

    def test_new_obj(self):
        strategy = DummyFilterStrategy()
        filter_prototype = Filter(strategy, history_size=5)
        tracker = Tracker(filter_prototype, distance_threshold=1.)

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
            tracker.update(state)
            tracked_state = tracker.get_tracked_objects()

            self.assertEqual(1, len(tracked_state))
            self.assertEqual(tracked_state[0].id, expected_tracking_id)

    def test_interpolate_object_with_ttl(self):
        strategy = PolynomialFilterStrategy(poly_degree=1, reject_outliers=False)
        filter_prototype = Filter(strategy, history_size=3)
        tracker = Tracker(filter_prototype, distance_threshold=1., max_time_to_live=2)

        states = [
            np.array([[1.0, 1.0]]),
            np.array([[1.5, 1.5]]),
            np.array([[2.0, 2.0]]),
            np.array([]),
            np.array([[3.0, 3.0]]),
            np.array([[3.5, 3.5]]),
            np.array([[4.0, 4.0]])
        ]
        
        expected_tracking_ids = [1, 1, 1, 1, 1, 1, 1]

        for i, (state, expected_tracking_id) in enumerate(zip(states, expected_tracking_ids)):
            print("timestep", i)
            tracker.update(state)
            tracked_state = tracker.get_tracked_objects()

            self.assertEqual(1, len(tracked_state))
            self.assertEqual(tracked_state[0].id, expected_tracking_id)