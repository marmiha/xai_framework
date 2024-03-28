from xai_framework.types import Dataset
from enum import Enum
import pandas as pd 
import os

# The saved datasets. Add the additional datasets here.
class DatasetFilename(Enum):
    # Real Datasets
    ADULT = "real/adult.csv"
    BOSTON_HOUSING = "real/boston_housing.csv"
    BREAST_CANCER = "real/breast_cancer.csv"
    FOREST_FIRES = "real/forest_fires.csv"
    IRIS = "real/iris.csv"
    TITANIC = "real/titanic.csv"

    #  Synthetic Datasets
    EXAMPLE_SYN = "synthetic/example_syn.csv"

    # Returns the target variable
    def target(self) -> str:
        return targets[self]

    def __str__(self) -> str:
        return self.value

# Target columns for each dataset
targets = {
    DatasetFilename.ADULT: "income",
    DatasetFilename.BOSTON_HOUSING: "MEDV",
    DatasetFilename.BREAST_CANCER: "Class",
    DatasetFilename.FOREST_FIRES: "area",
    DatasetFilename.IRIS: "Species",
    DatasetFilename.TITANIC: "Survived",
    DatasetFilename.EXAMPLE_SYN : "y"
}

class DatasetLoader:
    """
    A class for loading datasets.

    Attributes:
        dir (str): The directory where the dataset is located. Defaults to the DATASETS_DIR environment variable.

    Methods:
        load(filename: str) -> Dataset:
            Load the dataset with the given filename.
    """

    def __init__(self, dir: str) -> None:
        self.dir = dir
    
    def load(self, dataset_id: DatasetFilename) -> Dataset:
        """
        Load the dataset with the given filename from the directory.

        Args:
            name (DatasetId): The dataset to be loaded.

        Returns:
            Dataset: The loaded dataset.
        """
        df = pd.read_csv(os.path.join(self.dir, str(dataset_id)))
        X = df.drop(dataset_id.target(), axis=1).values
        y = df[dataset_id.target()].values
        feature_names = df.columns.drop(dataset_id.target()).tolist()

        return Dataset(
            name=dataset_id, 
            X=X, 
            y=y, 
            feature_names=feature_names
        )
