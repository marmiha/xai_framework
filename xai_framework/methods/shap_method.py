import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import shap
from xai_framework.types import ExplainabilityMethod

class ShapMethod(ExplainabilityMethod):
    name = "Shap"

    def explain(self, X: np.ndarray, model: BaseEstimator) -> np.ndarray:
        """
        Generate Shap values for the given input data.

        Parameters:
        - X: Input data for which explanations will be generated.
        - model: The model to be explained.

        Returns:
        - explanations: A pandas DataFrame containing the Shap values.
        """

        # TODO: Is this ok?
        masker = shap.maskers.Independent(X)
        explainer = shap.Explainer(model.predict, masker)

        print(explainer)
        return explainer(X).values
