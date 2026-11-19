from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

THRESHOLD = 0.5


def evaluate_model(model, X, y, threshold=THRESHOLD) -> dict:
    """Evaluate a fitted pipeline. Positive class: Churn = Yes (label 1).

    Average precision uses continuous positive-class scores. Precision, recall,
    F1 and the confusion matrix use class predictions at a fixed threshold.
    """
    classes = list(model.classes_)
    scores = model.predict_proba(X)[:, classes.index(1)]
    pred = (scores >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "threshold": threshold,
    }
