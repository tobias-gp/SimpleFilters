from unittest import TestCase

from ..simple_filters import Tracker, TrackedObject, Filter, PolynomialFilterStrategy

import pytest
import numpy as np

class TestTracker(TestCase):

    def setUp(self): 
        strategy = PolynomialFilterStrategy()
        filter_prototype = Filter(strategy, history_size=10)
        self.tracker = Tracker(filter_prototype, distance_threshold=1.0)

    def test_tracker_update_objects(self): 
        self.static_update_and_assert(2, 2, 2) 
        self.static_update_and_assert(2, 2, 2) 

    def test_tracker_new_objects(self): 
        self.static_update_and_assert(2, 2, 2) 
        self.static_update_and_assert(3, 3, 3) 

    def test_tracker_delete_objects(self): 
        self.static_update_and_assert(2, 2, 2)
        self.static_update_and_assert(1, 1, 2)

    def test_tracker_delete_objects_time_to_live(self): 
        self.tracker.time_to_live = 1

        self.static_update_and_assert(2, 2, 2)
        self.static_update_and_assert(1, 2, 2) # object should be retained, even if it doesn't appear
        self.static_update_and_assert(1, 1, 2) # object should be removed after this update

    def test_tracker_mapping(self): 
        # TODO: This assumes that the order is retained, but makes it easier for testing
        reference_matrix = np.array([[1., 1., 2.], [2., 2., 3.]])

        self.tracker.update(self.generate_static_states(2))
        self.assertTrue((self.tracker.to_numpy_array() == reference_matrix).all())

        self.tracker.update(self.generate_static_states(2))
        self.assertTrue((self.tracker.to_numpy_array() == reference_matrix).all())

    def static_update_and_assert(self, number_of_states, assert_number_of_tracked_objects, assert_object_counter): 
        self.tracker.update(self.generate_static_states(number_of_states, with_noise=True))
        self.assertEqual(len(self.tracker.get_tracked_objects()), assert_number_of_tracked_objects)
        self.assertEqual(self.tracker.object_counter, assert_object_counter)

    def generate_static_states(self, num_states, with_noise=False): 
        states = []
        for i in range(0, num_states):
            # generate some noise 
            if with_noise: 
                random_noise = (np.random.rand(2) / 2) - 0.5
            else:
                random_noise = 0.0

            state = np.array([i + 1, i + 2]) + random_noise
            states.append(state)

        return states