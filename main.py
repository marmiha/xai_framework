from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from xai_framework.methods.shap_method import ShapMethod
from xai_framework.methods.lime_method import LimeMethod
from xai_framework.methods.shapley_sampling_values_method import ShapleySampling
from xai_framework.methods.kernel_shap_method import KernelShap

from xai_framework.types import Dataset
from xai_framework.utils import process_feature_importance

from sklearn.datasets import load_iris

iris = load_iris()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(C=1, kernel="linear", probability=True))
])

dataset = Dataset(
    name="iris",
    X=iris.data,
    y=iris.target,
    feature_names=iris.feature_names
)
#print(dataset.y)

shap = ShapMethod()

lime = LimeMethod()
shapley_sampling = ShapleySampling()
kernel_shap = KernelShap()
importance = process_feature_importance(
    learner=model,
    dataset=dataset,
    methods=[
        lime, shap, shapley_sampling, kernel_shap
    ]
)



print(importance)

from xai_framework.writers.local_writer import LocalWriter

writer = LocalWriter(output_dir="./output")
writer.write(importance, "feature_importance.csv")

