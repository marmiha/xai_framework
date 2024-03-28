

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import shap
from xai_framework.types import ExplainabilityMethod

class ShapleySampling(ExplainabilityMethod):
    name = "Shapley sampling values"

    def explain(self, X: np.ndarray, y: np.ndarray, model: BaseEstimator, column_names=None, categorical_columns=None, prediction_type="None") -> np.ndarray:

        """
        Generate Shap values for the given input data.

        Parameters:
        - X: Input data for which explanations will be generated.
        - model: The model to be explaSined.

        Returns:
        - explanations: A pandas DataFrame containing the Shap values.
        """

        # TODO: Is this ok?
        #masker = shap.maskers.Independent(X)
        #print(masker)
        explainer = shap.SamplingExplainer(model.predict, X)
        print(explainer)
        print(type(explainer(X).values))
        return explainer(X).values
