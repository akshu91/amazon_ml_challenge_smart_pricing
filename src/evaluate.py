# src/evaluate.py
import numpy as np

def smape(y_true, y_pred):
    # y_true, y_pred: numpy arrays in original price-space (not log)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # avoid divide by zero, treat denom==0 -> set denom small
    denom = np.where(denom == 0, 1e-6, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0

def log_to_price(pred_log):
    import numpy as np
    return np.expm1(pred_log).clip(0.01, None)
