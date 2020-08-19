import numpy as np
from copy import deepcopy

from . import Filter

class TrackedObject(Filter): 
    """
    Acts as a simple proxy to the actual filter provided in the initialization
    """

    def __init__(self, id, filter): 
        self.id = id
        self.filter = filter
        self.time_to_live = 0

    def update(self, state): 
        self.filter.update(state)

    def eval(self, time=0): 
        return self.filter.eval(time=time)

    def raw(self, time=0): 
        return self.filter.raw(time=time)

    def get_history(self): 
        return self.filter.get_history()

class Tracker: 
    """
    Implements an multi-object tracker, that can use the given filters
    """

    def __init__(self, filter_prototype, time_to_live=0, distance_threshold=1.0): 
        self.distance_threshold = distance_threshold
        self.time_to_live = time_to_live
        self.object_counter = 0

        self.__filter_prototype = filter_prototype
        self.__tracked_objects = []

    def get_tracked_objects(self): 
        return self.__tracked_objects

    def to_numpy_array(self, raw=False): 
        """
        Returns the tracking id, plus the filtered object state if raw is False
        """
        m = []
        for t in self.__tracked_objects: 
            if raw: 
                state = t.raw()
            else: 
                state = t.eval()
            
            m.append(np.insert(state, 0, t.id))

        return np.array(m)

    def update(self, states):
        """
        Updates the list of tracked objects by mapping the closest objects to the new states. 
        Objects which cannot be mapped are either added or removed. 
        """

        states = np.array(states)

        # check if the states array is 2d, otherwise make it so
        if len(states.shape) == 1: 
            states = np.array([states])

        # check if the states array is empty
        if states.size == 0: 
            number_of_states = 0
        else: 
            number_of_states = states.shape[0]
        
        number_of_tracked_objects = len(self.__tracked_objects)
        objects_to_match = [i for i in range(0, number_of_tracked_objects)] 
        states_to_match = [i for i in range(0, number_of_states)]

        ## Build the distance matrix and match objects
        # We build a matrix that contains the distances of the tracked objects 
        # with its predicted state (determined by the filter) and the new states which just came in
        if number_of_tracked_objects > 0 and number_of_states > 0: 

            # Calculate the distance matrix, the complexity is n^2
            distance_matrix = np.zeros((number_of_tracked_objects, number_of_states))
            for t, tracked_object in enumerate(self.__tracked_objects):
                # get the predicted state of this object
                pred_state = tracked_object.eval(time=1)

                for s in range(0, number_of_states): 
                    state = states[s, :]
                    # TODO: Should this be moved to filter strategy?
                    distance_matrix[t, s] = np.linalg.norm(pred_state - state)

            # Now we match the tracked objects to the objects in the distance matrix 
            # Clearly, this is not the optimal solution (but fast), since it doesn't 
            # optimize for the cumulative distance of pairs
            # Whenever a minimum was found, we invalidate this part in the distance matrix
            min_distances = np.min(distance_matrix, axis=1)
            min_distances = np.sort(min_distances)
            
            for d in min_distances: 
                # when the distance threshold is exceeded, it means we are seeing new objects, 
                # or existing ones shall be removed
                if d > self.distance_threshold: 
                    break

                index = np.argwhere(distance_matrix == d)[0]
                t = index[0] # just to avoid confusion, this is the object
                s = index[1] # and this the state
                distance_matrix[t, :] = np.inf # invalidate this part of the distance matrix

                objects_to_match.remove(t)
                states_to_match.remove(s)
                self.__tracked_objects[t].update(states[s])

        ## Add objects
        # now go through all unmatched objects and create new objects
        for i in states_to_match: 
            self.object_counter += 1
            added_object = TrackedObject(self.object_counter, deepcopy(self.__filter_prototype))
            added_object.update(states[i])
            self.__tracked_objects.append(added_object)

        ## Delete objects
        # Remove an object that has not been seen when its time-to-live is exceeded
        for i in objects_to_match: 
            tracked_object = self.__tracked_objects[i]
            tracked_object.time_to_live += 1

            if tracked_object.time_to_live > self.time_to_live: 
                self.__tracked_objects.remove(tracked_object)
            else:  
                # update the object with the next predicted state
                tracked_object.update(tracked_object.eval(time=1))
