"""
Model evaluation utilities for TDA applications.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score


class ModelEvaluator:
    """Basic model evaluation utilities."""
    
    def __init__(self):
        pass
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Basic model evaluation."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted')
        }


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Convenience function for model evaluation."""
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(y_true, y_pred)
