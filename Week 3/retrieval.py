import faiss
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_retrieval(queries, features, metric='cosine', k=None):
    """Perform k-nearest neighbor retrieval using a specific metric.
    Args:
        queries: numpy array of shape (n_queries, n_features)
        features: numpy array of shape (n_images, n_features)
        k: number of nearest neighbors to retrieve
    Returns:
        indices: numpy array of shape (n_queries, k)
    """
    # Create a NearestNeighbors instance
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    # Fit the features
    nn.fit(features)
    # Perform the search
    distances, indices = nn.kneighbors(queries)
    return indices


def faiss_knn_retrieval(queries, features, k=5):
    """Perform k-nearest neighbor retrieval using Faiss.
    Args:
        queries: numpy array of shape (n_queries, n_features)
        features: numpy array of shape (n_images, n_features)
        k: number of nearest neighbors to retrieve
    Returns:
        indices: numpy array of shape (n_queries, k)
    """
    # Create a flat index
    index = faiss.IndexFlatL2(features.shape[1])
    # Add the features to the index
    index.add(np.ascontiguousarray(features, dtype=np.float32))
    # Perform the search
    distances, indices = index.search(np.ascontiguousarray(queries, dtype=np.float32), k=k)
    return indices


def retrieval(queries, features, method='knn', metric='cosine', k=None):
    """Perform k-nearest neighbor retrieval using a specific method and metric.
    Args:
        queries: numpy array of shape (n_queries, n_features)
        features: numpy array of shape (n_images, n_features)
        method: method to use, e.g. 'knn', 'faiss'
        metric: metric to use, e.g. 'cosine', 'euclidean'
        k: number of nearest neighbors to retrieve
    Returns:
        indices: numpy array of shape (n_queries, k)
    """
    # Last column contains the labels
    new_features = features[:, :-1]
    new_queries = queries[:, :-1]
    if k is None:
        k = features.shape[0]
    if method == 'knn':
        indices = knn_retrieval(new_queries, new_features, metric, k)
    elif method == 'faiss':
        indices = faiss_knn_retrieval(new_queries, new_features, k)
    else:
        raise ValueError(f"Method {method} not available")

    # Create a new array with the labels that correspond to the retrieved indices
    labels = np.zeros((indices.shape[0], k))
    for i in range(indices.shape[0]):
        labels[i, :] = features[indices[i], -1]
    return indices, labels