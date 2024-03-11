from enum import Enum
import numpy as np
import pandas as pd

from xai_framework.synthetic_data_generator.syege import generate_syntetic_rule_based_classifier, generate_synthetic_linear_classifier
from xai_framework.types import Dataset


class DatasetGeneratorType(Enum):
    RuleBased = 'RuleBased'
    Linear = 'Linear'


class DatasetGenerator:
    def __init__(
        self,
        type: DatasetGeneratorType,
        n_features: int, n_samples: int,
        dataset_name: str,
        n_irrelevant_features=0,
        expression=None,
        random_state=None,
        noise_level_rb=None,
        save=False
    ):
        self.type = type
        self.n_features = n_features
        self.n_samples = n_samples
        self.dataset_name = dataset_name
        self.n_irrelevant_features = n_irrelevant_features
        self.expression = expression
        self.random_state = random_state
        self.noise_level_rb = noise_level_rb
        self.save = save

    def generate_dataset(self):
        match self.type:
            case DatasetGeneratorType.RuleBased:
                sc = generate_syntetic_rule_based_classifier(
                    n_samples=self.n_samples,
                    n_features=self.n_features,
                    n_all_features=(self.n_features +
                                    self.n_irrelevant_features),
                    random_state=self.random_state,
                    factor=self.noise_level_rb
                )
            case DatasetGeneratorType.Linear:
                sc = generate_synthetic_linear_classifier(
                    expr=None,
                    n_features=self.n_features,
                    n_all_features=(self.n_features +
                                    self.n_irrelevant_features),
                    random_state=self.random_state,
                    n_samples=self.n_samples
                )
            case _:
                raise ValueError(f"Invalid type. Supported types are: {[e.value for e in DatasetGeneratorType]}")

        dataset = Dataset(
            name=self.dataset_name,
            X=sc["X"],
            y=sc["Y"],
            feature_names=sc["feature_names"]
        )

        if self.save:
            path = f"../datasets/synthetic/{self.dataset_name}.csv"

            df = pd.DataFrame(sc["X"], columns=sc["feature_names"])
            df["y"] = sc["Y"]

            df.to_csv(path, index=False)

        return dataset


if __name__ == "__main__":
    # For testing purposes.
    generator = DatasetGenerator(
        type=DatasetGeneratorType.RuleBased,
        n_features=5,
        n_irrelevant_features=0,
        n_samples=500,
        dataset_name="syn_ds",
        noise_level_rb=2,
        save=True
    )
    ds = generator.generate_dataset()
