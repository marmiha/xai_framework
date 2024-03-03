import numpy as np
import pandas as pd

from xai_framework.synthetic_data_generator.syege import generate_syntetic_rule_based_classifier, generate_synthetic_linear_classifier

from xai_framework.types import Dataset

class DatasetGenerator:
    def __init__(self, type, n_features, n_samples, dataset_name, n_irrelevant_features=0, expression=None, random_state=None, noise_level_rb=None, save=False):
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
        if self.type == 'rule_based':
            sc = generate_syntetic_rule_based_classifier(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_all_features=(self.n_features + self.n_irrelevant_features),
                random_state=self.random_state,
                factor=self.noise_level_rb
            )
        elif self.type == 'linear':
            sc = generate_synthetic_linear_classifier(
                expr=None, 
                n_features=self.n_features,
                n_all_features=(self.n_features + self.n_irrelevant_features), 
                random_state=self.random_state,
                n_samples=self.n_samples
            )
        else:
            raise ValueError("Invalid type. Supported types are 'rule_based' and 'linear'.")

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

    generator = DatasetGenerator(
        type="rule_based",
        n_features=5,
        n_irrelevant_features=0,
        n_samples=500,
        dataset_name="syn_ds",
        noise_level_rb=2,
        save=True
    )
    ds = generator.generate_dataset()