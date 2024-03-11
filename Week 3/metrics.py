import numpy as np
from sklearn.metrics import average_precision_score


def precision_at_k(retrieved, gt, k):
    """Compute the precision at k.
    Args:
        retrieved: numpy array of shape (n_queries, k)
        gt: numpy array of shape (n_queries, n_relevant)
        k: number of retrieved items
    Returns:
        precision: numpy array of shape (n_queries,)
    """
    precision = np.zeros(retrieved.shape[0])
    for i in range(retrieved.shape[0]):
        precision[i] = len(np.intersect1d(retrieved[i, :k], gt[i])) / k
    return precision


def recall_at_k(retrieved, gt, k):
    """Compute the recall at k.
    Args:
        retrieved: numpy array of shape (n_queries, k)
        gt: numpy array of shape (n_queries, n_relevant)
        k: number of retrieved items
    Returns:
        recall: numpy array of shape (n_queries,)
    """
    recall = np.zeros(retrieved.shape[0])
    for i in range(retrieved.shape[0]):
        recall[i] = len(np.intersect1d(retrieved[i, :k], gt[i])) / len(gt[i])
    return recall


def average_precision()