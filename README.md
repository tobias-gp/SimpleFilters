# Simple Filters

This is a collection of simple filters to filter a multi-dimensional time series with real-time performance. 

## Usage

Set up your filter: 
```
from filter import Filter, PolynomialFilterStrategy

strategy = PolynomialFilterStrategy(update_rate_in_ms=100, poly_order=3, predict_samples=2)
self.filter = Filter(strategy, history_size=10)
```

Fill the history: 
```
self.filter.update_state([x, y, z, rotation])
```

Get the predictions: 
```
result = self.filter.get_filtered_state()
```

## Test

Simply run ```pytest``` in the project directory. 