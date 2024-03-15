import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, auc


def precision_at_k(binary_results):
    """Compute the precision at k.
    Args:
        binary_result: numpy array of shape (n_samples, n_retrieved)
        k: number of retrieved items to consider
    Returns:
        precision: numpy array of shape (n_samples) containing the precision at k for each query
    """
    cumsums = np.cumsum(binary_results, axis=1)
    precisions = cumsums / np.arange(1, binary_results.shape[1] + 1)
    return precisions


def recall_at_k(binary_results):
    """Compute the recall at k.
    Args:
        binary_result: numpy array of shape (n_samples, n_retrieved)
        k: number of retrieved items to consider
    Returns:
        recall: numpy array of shape (n_samples) containing the recall at k for each query
    """
    cumsums = np.cumsum(binary_results, axis=1)
    recalls = cumsums / np.sum(binary_results, axis=1).reshape(-1, 1)
    return recalls


def average_precision(binary_results, k=None):
    """Compute the average precision.
    Args:
        binary_result: numpy array of shape (n_samples, n_retrieved)
    Returns:
        average_precision: float
    """
    br = binary_results.copy()
    if k is not None:
        br = binary_results[:, :k]
    cum_sum = np.cumsum(br, axis=1)
    # Divide by the column index + 1
    precisions = cum_sum / np.arange(1, br.shape[1] + 1)
    # Multiply by the binary results
    m = precisions * br
    n = np.sum(m, axis=1)
    total_docs = np.sum(binary_results, axis=1)
    return n / total_docs


def plot_precision_recall_curve(precision, recall):
    """Plot the precision-recall curve.
    Args:
        precision: numpy array of shape (n_samples)
        recall: numpy array of shape (n_samples)
    """
    plt.plot(recall, precision, marker='.')
    # Change limits of the plot to 0-1
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # Calculate the area under the curve
    area = auc(recall, precision)
    # Add the title and the legend
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.text(0.6, 0.1, f'AUC = {area:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.show()
