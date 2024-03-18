from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xai_framework.methods.shap_method import ShapMethod
from xai_framework.types import Dataset
from xai_framework.utils import process_feature_importance
from xai_framework.evaluation_metrics.utils import evaluate_explanation
from xai_framework.evaluation_metrics.similarity import CorrelationOfFeatureImportance, CosineSimilarity, EuclideanDistance, TopKRankingMatch

from sklearn.datasets import load_iris

iris = load_iris()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearSVC(C=1, loss="hinge"))
])

dataset = Dataset(
    name="breast_cancer",
    X=iris.data,
    y=iris.target,
    feature_names=iris.feature_names
)

shap = ShapMethod()

importance = process_feature_importance(
    learner=model,
    dataset=dataset,
    methods=[
        shap
    ]
)

print(importance)

# to-do we need to formulate a unified way to present the importance(explanation)
for explanation in importance:
    ground_truth = None # to-do I am not sure where the ground truth should come from
    print(evaluate_explanation(explanation, groud_truth, CorrelationOfFeatureImportance()))
    print(evaluate_explanation(explanation, groud_truth, CosineSimilarity()))
    print(evaluate_explanation(explanation, groud_truth, EuclideanDistance()))
    print(evaluate_explanation(explanation, groud_truth, TopKRankingMatch(k=2)))


from xai_framework.writers.local_writer import LocalWriter

writer = LocalWriter(output_dir="./output")
writer.write(importance, "feature_importance.csv")

