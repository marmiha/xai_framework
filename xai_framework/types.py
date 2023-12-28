from sklearn.base import BaseEstimator
import numpy as np 

class Dataset:
    def __init__(self, name: str, X: np.ndarray, y: np.ndarray, feature_names: list[str] = None):
        self.X = X
        self.y = y
        self.feature_names = feature_names

    def __str__(self):
        return self.X

    def __repr__(self):
        return self.X

class ExplainabilityMethod:
    name: str = None

    def explain(self, X: np.ndarray, model: BaseEstimator) -> np.ndarray:
        """
        Generate explanations for the given input data.

        Parameters:
        - X: Input data for which explanations will be generated.
        - model: The model to be explained.

        Returns:
        - explanations: A pandas DataFrame containing the explanations.
        """
        raise NotImplementedError("The 'explain' method must be implemented in subclasses.")

    def __str__(self):
        return self.name
