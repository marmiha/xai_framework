from sklearn.base import BaseEstimator
import numpy as np 
from abc import ABC
import pandas as pd

class Dataset:
    """
    Represents a dataset with input features and corresponding labels.

    Attributes:
        name (str): The name of the dataset.
        X (np.ndarray): The input features of the dataset.
        y (np.ndarray): The labels of the dataset.
        feature_names (list[str], optional): The names of the features.

    Methods:
        __str__(): Returns a string representation of the dataset.
    """

    def __init__(self, name: str, X: np.ndarray, y: np.ndarray, feature_names: list[str] = None):
        self.name = name
        self.X = X
        self.y = y
        self.feature_names = feature_names

    def __str__(self):
        return str(self.X)


class ExplainabilityMethod(ABC):
    """
    Base class for explainability methods.

    Attributes:
    - name: The name of the explainability method.

    Methods:
    - explain(X: np.ndarray, model: BaseEstimator) -> np.ndarray:
        Generate explanations for the given input data.

    - __str__() -> str:
        Return the name of the explainability method as a string.
    """
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

class Writer(ABC):
    """
    Base class for dataframe writers.

    Attributes:
    - name: The name of the writer.

    Methods:
    - write(data: dict):
        Write the given data to the output.

    - __str__() -> str:
        Return the name of the writer as a string.
    """
    name: str = None

    def __init__(self, name: str):
        self.name = name
        
    def write(self, df: pd.DataFrame):
        """
        Write the df to the output.

        Parameters:
        - df: The dataframe to be written.
        """
        raise NotImplementedError("The 'write' method must be implemented in subclasses.")

    def __str__(self):
        return self.name
