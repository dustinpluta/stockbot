from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)


def sensitivity(y_true, y_pred):
    """Also known as recall or true positive rate."""
    return recall_score(y_true, y_pred)


def specificity(y_true, y_pred):
    """True negative rate: TN / (TN + FP)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


METRIC_REGISTRY = {
    "accuracy": accuracy,
    "f1": f1,
    "sensitivity": sensitivity,
    "specificity": specificity,
}
