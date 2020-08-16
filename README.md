# Simple Filters and Tracking for Real-time Time-Series Data

![Python package](https://github.com/tobias-gp/SimpleFilters/workflows/Python%20package/badge.svg)

This is a collection of simple filters optimized for real-time performance. 

## Filter

A filter acts as a container for a time-series. The length of the time-series is kept constant after initial filling, according to the ```history_size``` specified. 

Currently, two filters are implemented: 
* **MeanFilterStrategy**: Applies a simple mean to all values along axis 0
* **PolynomialFilterStrategy**: Returns the filtered last item (and optionally predicts the next item) of a multi-dimensional time series using a polynomial regression. The strategy can be applied to sensor data to retain smoothness while ensuring low latency and avoiding offsets with outliers. 

Set up your filter: 
```
from simple_filters import Filter, PolynomialFilterStrategy

strategy = PolynomialFilterStrategy(poly_degree=3, outlier_rejection_ratio=2.0)
filter = Filter(strategy, history_size=10)
```

Fill the history with a 1D array: 
```
filter.update([x, y, z, rotation])
```

Get the last filtered item at the current_time, or by specifying a time step in the future (```time=1```). 
```
result_current = filter.eval()
result_future = filter.eval(time=1)
```

## Tracker

Oftentimes, multiple objects must be tracked that also require filtering. SimpleFilters implements a simple multi-object tracker for this purpose. 

The following properties can be defined: 
* **distance_threshold**: Maximum distance to match objects - when the threshold is exceeded, a new object will be created 
* **time_to_live**: If an object is not seen, it is still retained for the given number of state updates
* **filter_prototype**: A filter that will be cloned for each new appearing object

Set up your tracker: 
```
from simple_filters import Filter, PolynomialFilterStrategy, Tracker

strategy = PolynomialFilterStrategy(poly_degree=3, outlier_rejection_ratio=2.0)
filter_prototype = Filter(strategy, history_size=10)
tracker = Tracker(filter_prototype, distance_threshold=1.0, time_to_live=1)
```

Update your tracker with a 2D array: 
```
tracker.update([[1.0, 1.0], [2.0, 2.0]])
tracker.update([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
```

Retrieve the results: 
```
# Access the list of objects like any filter: 
list_of_objects = tracker.get_tracked_objects() 
list_of_objects[0].eval(0) # get the latest state ([1.0, 1.0])
list_of_objects[0].id # get the tracking ID

# Or convert to a NumPy array: 
np_array = tracker.to_numpy_array()
```

## Testing

Simply run ```pytest``` in the project directory. 
