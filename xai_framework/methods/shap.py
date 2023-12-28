import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import shap
from xai_framework.types import ExplainabilityMethod

class ShapExplainabilityMethod(ExplainabilityMethod):
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

        # Initialize the Shap explainer with the model's predict function
        explainer = shap.Explainer(model)

        # Generate Shap values for the input data
        shap_values = explainer(X)

        # Convert Shap values to a pandas DataFrame
        explanations = pd.DataFrame(shap_values, columns=[f'shap_value_{i}' for i in range(X.shape[1])])

        return explanations
