# Simple Time-Series Filters

![Python package](https://github.com/tobias-gp/SimpleFilters/workflows/Python%20package/badge.svg)

This is a collection of simple filters optimized for real-time performance. The history of filtered values is kept constant after the initial filling according to the ```history_size``` specified. 

## Usage

Set up your filter: 
```
from simple_filters import Filter, PolynomialFilterStrategy

strategy = PolynomialFilterStrategy(poly_degree=3, outlier_rejection_ratio=2.0)
filter = Filter(strategy, history_size=10)
```

Fill the history: 
```
filter.update([x, y, z, rotation])
```

Get the last filtered item at the current_time, or by specifying a time step in the future (```time=1```). 
```
result_current = filter.eval()
result_future = filter.eval(time=1)
```

## Filter Strategies

Currently, only few filters are implemented: 
* **MeanFilterStrategy**: Applies a simple mean to all values along axis 0
* **PolynomialFilterStrategy**: Returns the filtered last item (and optionally predicts the next item) of a multi-dimensional time series using a polynomial regression. The strategy can be applied to sensor data to retain smoothness while ensuring low latency and avoiding offsets with outliers. 

## Test

Simply run ```pytest``` in the project directory. 
