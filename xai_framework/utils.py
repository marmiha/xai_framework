from sklearn.base import BaseEstimator
from xai_framework.types import ExplainabilityMethod, Dataset
import pandas as pd
from joblib import Parallel, delayed

def process_feature_importance(learner: BaseEstimator, dataset: Dataset, methods: list[ExplainabilityMethod]) -> list[pd.DataFrame]:
    """
    Process feature importance for a given learner and dataset using multiple explainability methods.

    Args:
        learner (BaseEstimator): The machine learning model to train.
        dataset (Dataset): The dataset containing the features and target variable.
        methods (list[ExplainabilityMethod]): List of explainability methods to use.

    Returns:
        list[pd.DataFrame]: List of dataframes containing the feature importances for each method.
    """
    # TODO: Cross-Validation
    learner.fit(dataset.X, dataset.y)

    df_dict = {feature: [] for feature in dataset.feature_names}
    df_dict["method"] = []
    df = pd.DataFrame(df_dict)

    # Generate explanations for each method in parallel
    def generate_explanations(method: ExplainabilityMethod) -> tuple[ExplainabilityMethod, pd.DataFrame]:
        print("Generating explanations for method: " + str(method))
        explanations = method.explain(dataset.X, model=learner)
        return (method, explanations)

    # Run in parallel
    job_results = Parallel(n_jobs=-1)(delayed(generate_explanations)(method) for method in methods)

    for method, explanations in job_results:
       # Convert explanations numpy matrix to any type 
       for row in explanations:
            # TODO: Maybe a really slow operation?
            entry = list(row) + [str(method)]
            df.loc[len(df)] = entry

    return df
