import os
import torch
import numpy as np
from datasets import load_dataset
from models import ResNet50, SiameseNet, TripletNet


def feature_extraction(tag, dataset, model, output_folder):
    """Extract features from a set of images using a pre-trained model.
    Features and labels will be stored in a .npy file in the format: [features, labels].
    For each image in the dataset, we have a row of features (e.g. 2048 for ResNet50) and a label.
    """
    features = np.array([])
    labels = np.array([])
    model.eval()
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


def perform_feature_extraction(path_to_data, data, model_path, output_path):
    """
    Extract features from the dataset using a pre-trained model
    :param data: Folder containing the dataset
    :param model_path: Path to the pre-trained model
    :param output_path: Output path to store the features and labels
    :return:
    """
    dataset = path_to_data + data
    train_data = load_dataset(dataset + "/train", 32, False)
    validation_data = load_dataset(dataset + "/test", 32, False)

    # Load the model
    if model_path == "None":
        model_name = "ResNet50"
        model = ResNet50()
    elif "siamese" in model_path:
        model = SiameseNet(ResNet50())
        # Load weights
        model.load_state_dict(torch.load('siamese_model.pth'))
        model = model.embedding_net
        model_name = "SiameseNet"
    elif "triplet" in model_path:
        model = TripletNet(ResNet50())
        # Load weights
        model.load_state_dict(torch.load('models/triplet_model.pth'))
        model = model.embedding_net
        model_name = "TripletNet"
    else:
        raise ValueError(f"Model {model_path} not available")

    f_output_folder = f"{output_path}/{model_name}/{data}/"
    feature_extraction("train", train_data, model, f_output_folder)
    feature_extraction("validation", validation_data, model, f_output_folder)
