

import numpy as np
from sklearn.base import BaseEstimator
import shap
from xai_framework.types import ExplainabilityMethod

class KernelShap(ExplainabilityMethod):
    name = "Kernel shap"

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
        if prediction_type == "regression":
            predict_f = model.predict
        else:
            #prediction_f = lamda(x: model.predict_proba(x))
            prediction_f = model.predict_proba


        explainer = shap.KernelExplainer(prediction_f, X)
        #print(explainer)
        #print((explainer(X).values))
        #foo = explainer.explain(X)
        print(X.shape)
        foo = np.array(explainer.shap_values(X))


        results = []
        for i  in range(len(y)):
            results.append(foo[y[i], i, :])




        return np.array(results)