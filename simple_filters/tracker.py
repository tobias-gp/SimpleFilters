import numpy as np
from copy import deepcopy

from scipy.optimize import linear_sum_assignment

from . import Filter

class TrackedObject(Filter): 
    """
    Acts as a simple proxy to the actual filter provided in the initialization
    """

    def __init__(self, id, filter, max_time_to_live, time_to_birth): 
        self.id = id
        self.filter = filter
        self.time_to_live = 1
        self.max_time_to_live = max_time_to_live
        self.time_to_birth = time_to_birth
        self.is_born = (self.time_to_birth <= self.time_to_live)

    def increase_time_to_live(self): 
        if self.time_to_live < self.max_time_to_live: 
            self.time_to_live += 1

        if not self.is_born: 
            self.is_born = (self.time_to_birth <= self.time_to_live)

    def decrease_time_to_live(self): 
        if self.time_to_live > 0: 
            self.time_to_live -= 1
        else:
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

    def __init__(self, filter_prototype, 
                    max_time_to_live=1, 
                    time_to_birth=1,
                    distance_threshold=1.0, 
                    distance_function=lambda x1, x2: np.linalg.norm(x1 - x2)): 
        self.distance_threshold = distance_threshold
        self.max_time_to_live = max_time_to_live
        self.time_to_birth = time_to_birth

        self.object_counter = 0

        self.__distance_function = distance_function
        self.__filter_prototype = filter_prototype
        self.__tracked_objects = []

    def get_tracked_objects(self): 
        return list(filter(lambda x: x.is_born, self.__tracked_objects))

    def to_numpy_array(self, raw=False): 
        """
        Returns the tracking id, plus the filtered object state if raw is False
        """
        m = []
        for t in self.get_tracked_objects(): 
            if raw: 
                state = t.raw()
            else: 
                state = t.eval()
            
            m.append(np.insert(np.array(t.id, dtype=np.float32), 0, state))

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
        objects_to_match = list([i for i in range(0, number_of_tracked_objects)])
        states_to_match = list([i for i in range(0, number_of_states)])

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
                    distance_matrix[t, s] = self.__distance_function(pred_state, state)

            # Now we match the tracked objects to the objects in the distance matrix 
            # We do this by applying minimum weight matching in bipartite graphs: 
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html 
            object_indices, state_indices = linear_sum_assignment(distance_matrix)

            for (t, s) in zip(object_indices, state_indices): 
                d = distance_matrix[t, s]
                
                if d > self.distance_threshold: 
                    continue

                objects_to_match.remove(t)
                states_to_match.remove(s)
                self.__tracked_objects[t].increase_time_to_live()
                self.__tracked_objects[t].update(states[s])

        ## Delete objects
        # Remove an object that has not been seen when its time-to-live is exceeded
        # We are using two steps for deletion, because in the first step we are addressing by index and we 
        # don't want to mess up the list indexing
        removals = []
        for i in objects_to_match: 
            tracked_object = self.__tracked_objects[i]
            tracked_object.decrease_time_to_live()

            if tracked_object.time_to_live < 1: 
                removals.append(tracked_object)
            else:  
                # update the object with the next predicted state
                tracked_object.update(tracked_object.eval(time=1))

        for tracked_object in removals: 
            self.__tracked_objects.remove(tracked_object)

        ## Add objects
        # now go through all unmatched objects and create new objects
        for i in states_to_match: 
            self.object_counter += 1
            added_object = TrackedObject(
                                self.object_counter, 
                                deepcopy(self.__filter_prototype), 
                                max_time_to_live=self.max_time_to_live,
                                time_to_birth=self.time_to_birth
                            )
            added_object.update(states[i])
            self.__tracked_objects.append(added_object)
