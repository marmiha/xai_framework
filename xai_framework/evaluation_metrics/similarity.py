import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

from xai_framework.types import EvaluationMetric

class CorrelationOfFeatureImportance(EvaluationMetric):
    name = "Correlation of Feature Importance"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluate the given predictions.

        Parameters:
        - y_true: The true labels.
        - y_pred: The predicted labels.

        Returns:
        - score: The evaluation score.
        """
        return np.corrcoef(y_true, y_pred)
    
class CosineSimilarity(EvaluationMetric):
    name = "Cosine Similarity"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluate the given predictions.

        Parameters:
        - y_true: The true labels.
        - y_pred: The predicted labels.

        Returns:
        - score: The evaluation score.
        """
        return cosine_similarity(y_true, y_pred)
    
    
class EuclidianDistance(EvaluationMetric):
    name = "Euclidian Distance"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluate the given predictions.

        Parameters:
        - y_true: The true labels.
        - y_pred: The predicted labels.

        Returns:
        - score: The evaluation score.
        """
        return euclidean_distances(y_true, y_pred)
    
class TopKRankingMatch(EvaluationMetric):
    name = "Top K Ranking Match"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, k = 1) -> float:
        """
        Evaluate the given predictions.

        Parameters:
        - y_true: The true labels.
        - y_pred: The predicted labels.

        Returns:
        - score: The evaluation score.
        """
        return set(y_true[:k]) == set(y_pred[:k])
    
class MinimumEditDistanceOfRanking(EvaluationMetric):
    name = "Minimum-Edit Distance of Ranking"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluate the minimum edit distance (Levenshtein distance) needed to match the rankings.

        Parameters:
        - y_true: The true ranking.
        - y_pred: The predicted ranking.

        Returns:
        - score: The minimum edit distance.
        """
        m, n = len(y_true), len(y_pred)
        
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if y_true[i - 1] == y_pred[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],
                                       dp[i][j - 1],   
                                       dp[i - 1][j - 1]
                                      )
        return dp[m][n]
