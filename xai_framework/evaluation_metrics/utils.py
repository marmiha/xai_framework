def evaluate_explanation(y_true, y_pred, metrics):
    """
    Evaluate the metrics for the given y_true and y_pred
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param metrics: List of metrics to evaluate
    :return: Dictionary of metrics and their values
    """
    metric_values = {}
    for metric in metrics:
        metric_values[metric] = metric(y_true, y_pred)
    return metric_values