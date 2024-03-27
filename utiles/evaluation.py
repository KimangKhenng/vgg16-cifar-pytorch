from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def evaluate_classification(y_true, y_pred):
    """
    Evaluate classification model performance using common metrics.

    Args:
    - y_true (numpy array): True labels.
    - y_pred (numpy array): Predicted labels.

    Returns:
    - metrics_dict (dict): Dictionary containing evaluation metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

    return metrics_dict


# Example usage:
if __name__ == "__main__":
    # Generate example true and predicted labels
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1])

    # Evaluate classification model performance
    metrics = evaluate_classification(y_true, y_pred)

    # Print evaluation metrics
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
