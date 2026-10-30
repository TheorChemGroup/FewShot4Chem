"""Classes for validation by evaluating some metrics related to target prediction."""

from validation.target.classification import ClassificationValidator
from validation.target.regression import RegressionValidator

__all__ = ["ClassificationValidator", "RegressionValidator"]
