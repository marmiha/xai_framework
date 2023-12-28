from sklearn.base import BaseEstimator
from xai_framework.types import ExplainabilityMethod, Dataset
import pandas as pd
from joblib import Parallel, delayed

def process_feature_importance(learner: BaseEstimator, dataset: Dataset, methods: list[ExplainabilityMethod]) -> list[pd.DataFrame]:
    # Train the model
    learner.fit(dataset.X, dataset.y)

    df_dict = {feature: [] for feature in dataset.feature_names}
    df_dict["method"] = []

    # Generate explanations for each method in parallel
    def generate_explanations(method):
        importance = method.explain(dataset.X, model=learner)
        return [(feature, value) for feature, value in zip(dataset.feature_names, importance)]

    explanations = Parallel(n_jobs=-1)(delayed(generate_explanations)(method) for method in methods)

    # Collect the explanations
    for method, explanation in zip(methods, explanations):
        for feature, value in explanation:
            df_dict[feature].append(value)
        df_dict["method"].append(str(method))

    # Return the importances
    return pd.DataFrame(df_dict)
