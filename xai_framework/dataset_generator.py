import numpy as np

from synthetic_data_generator.syege import generate_syntetic_rule_based_classifier, generate_synthetic_linear_classifier

from xai_framework.types import Dataset

class DatasetGenerator:
    def __init__(self, type, n_features, n_samples, dataset_name, n_irrelevant_features=0, expression=None, random_state=None):
        self.type = type
        self.n_features = n_features
        self.n_samples = n_samples
        self.dataset_name = dataset_name
        self.n_irrelevant_features = n_irrelevant_features
        self.expression = expression
        self.random_state = random_state
        
    def generate_dataset(self):
        if self.type == 'rule_based':
            sc = generate_syntetic_rule_based_classifier(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_all_features=(self.n_features + self.n_irrelevant_features),
                random_state=self.random_state
            )
        elif self.type == 'linear':
            sc = generate_synthetic_linear_classifier(
                expr=None, 
                n_features=self.n_features,
                n_all_features=(self.n_features + self.n_irrelevant_feautures), 
                random_state=self.random_state,
                n_samples=self.n_samples
            )
        else:
            raise ValueError("Invalid type. Supported types are 'rule_based' and 'linear'.")

        dataset = Dataset(
            name=self.dataset_name,
            X=sc.X,
            y=sc.Y,
            feature_names=sc.feature_names
        )
        
        return dataset
