import os
import torch
import numpy as np


def feature_extraction(tag, dataset, model, output_folder):
    """Extract features from a set of images using a pre-trained model.
    Features and labels will be stored in a .npy file
    """
    features = np.array([])
    labels = np.array([])
    for inputs, labels_batch in dataset:
        with torch.no_grad():
            outputs = model(inputs).squeeze().numpy()
        features = np.append(features, outputs, axis=0) if features.size else outputs
        # Add labels as the last column in the features array
        labels = np.append(labels, labels_batch.numpy())
    # Add the labels as the last column in the features array
    labels = labels.reshape(-1, 1)
    features = np.append(features, labels, axis=1)
    # Check if output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    np.save(f"{output_folder}/{tag}_features_and_labels.npy", features)
    return features
