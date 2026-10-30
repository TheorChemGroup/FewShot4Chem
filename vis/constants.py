"""Constants used in visualization."""

from sklearn.metrics import mean_squared_error

MAX_ITEMS_FOR_DIMENSION_REDUCTION = 10000

MAX_ITEMS_FOR_METRICS = 10000

OUTLIER_FILTER = {  # IMPROVE: move somewhere else
    "metric": mean_squared_error,
    "name": "MSE",
    "outlier_fraction": 0.02,
}


MAX_ITEMS_FOR_ATOMIC_EMBEDDINGS = 10000

CLASSES_NAMES_FOR_BINARY_CLASSIFICATION = ["Positive", "Negative"]
