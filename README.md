# Simple Time-Series Filters

This is a collection of simple filters optimized for real-time performance. The history of filtered values is kept constant after the initial filling according to the ```history_size``` specified. 

## Usage

Set up your filter: 
```
from simple_filters import Filter, PolynomialFilterStrategy

strategy = PolynomialFilterStrategy(poly_degree=3, predict_samples=2, outlier_rejection_ratio=2.0)
filter = Filter(strategy, history_size=10)
```

Fill the history: 
```
filter.update_state([x, y, z, rotation])
```

Get the last filtered item, plus future predictions specified by predict_samples: 
```
result = filter.get_filtered_state()
```

## Filter Strategies

Currently, only few filters are implemented: 
* **MeanFilterStrategy**: Applies a simple mean to all values along axis 0
* **PolynomialFilterStrategy**: Returns the filtered last item (and optionally predicts the next item) of a multi-dimensional time series using a polynomial regression. The strategy can be applied to sensor data to retain smoothness while ensuring low latency and avoiding offsets with outliers. 

## Test

Simply run ```pytest``` in the project directory. 