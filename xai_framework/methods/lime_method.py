import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from xai_framework.types import ExplainabilityMethod
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import LimeTabular
import time

class LimeMethod(ExplainabilityMethod):
    name = "lime"

    def explain(self, X: np.ndarray, y: np.ndarray,  model: BaseEstimator, column_names=None, categorical_columns=None, prediction_type="None") -> np.ndarray:
        """
        Generate Lime values for the given input data.

        Parameters:
        - X: Input data for which explanations will be generated.
        - model: The model to be explained.
        - columns_names: List of names of column
        - prediction_type: "classification"/"regression"


        Returns:
        - explanations: A pandas DataFrame containing the Shap values.

        """
        #this can be in some configuration. for each dataset its own case in json file
        data = np.c_[X, y]
        column_names = column_names + ["label"]
        tabular_data = Tabular(
            data= data,
            categorical_columns=categorical_columns,
            feature_columns=column_names,
            target_column='label'
        )

        transformer = TabularTransform().fit(tabular_data)
        x = transformer.transform(tabular_data)

        # TODO: Is this ok?
        if prediction_type == "regression":
            predict_function = lambda z: model.predict(transformer.transform(z))

        else:
            predict_function = lambda z: model.predict_proba(transformer.transform(z))

        explainer = LimeTabular(
            training_data=tabular_data,
            predict_function=predict_function
        )
        test_instances = transformer.invert(x)

        print(explainer)

        explanations = explainer.explain(test_instances)

        print(explanations.explanations[0])

        print(column_names)
        column_names = column_names[:-1] #discard label
        mapp = {column_names[i]: i for i, f in enumerate(column_names)}

        results = []
        for sample in explanations.explanations:
            z = list(zip(map(lambda x: mapp[x], sample["features"]), sample["scores"]))
            z = sorted(z)
            z = [foo[1] for foo in z]
            results.append(z)
            #print(z)
            #time.sleep(1)

        return np.array(results) #need preprocessing