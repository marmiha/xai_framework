from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xai_framework.methods.shap import ShapExplainabilityMethod
from xai_framework.types import Dataset
from xai_framework.utils import process_feature_importance

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

shap = ShapExplainabilityMethod()

process_feature_importance(
    learner=model,
    dataset=dataset,
    methods=[
        shap
    ]
)
