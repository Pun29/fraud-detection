# src/models/baseline.py
import numpy as np
from src.evaluation.metrics import evaluate, log_result, EvalResult

def run_majority_class_baseline(y_val, y_test) -> EvalResult:
    """Always predicts 0 (legitimate). Score = 0 for all."""
    y_score = np.zeros(len(y_test))
    result = evaluate(y_test.values if hasattr(y_test, 'values') else y_test,
                      y_score, model_name="majority_class_baseline")
    log_result(result)
    return result
