import os
import torch
import numpy as np
from datasets import load_dataset
from models import ResNet50, SiameseNet, TripletNet, FasterRCNN


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
            outputs = model(inputs.cuda()).cpu().numpy()
        features = np.append(features, outputs, axis=0) if features.size else outputs
        # Add labels as the last column in the features array
        labels = np.append(labels, labels_batch)
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

    if "COCO" in dataset:
        train_data = load_dataset(dataset + "/train2014", 32, False)
        validation_data = load_dataset(dataset + "/val2014", 32, False)
    else:
        train_data = load_dataset(dataset + "/train", 32, False)
        validation_data = load_dataset(dataset + "/test", 32, False)

    # Load the model
    if model_path == "None":
        model_name = "ResNet50"
        model = FasterRCNN().cuda()
    else:
        try:
            model = torch.load(model_path)
            if "online" not in model_path:
                model = model.embedding_net
            model.cuda()
        except FileNotFoundError:
            print("Model not found. Please check the path to the model.")
            return
        # Get model name from the path (e.g. if path is models/online_model.pth, model_name = online_model)
        model_name = model_path.split("/")[-1].split(".")[0]

    f_output_folder = f"{output_path}/{model_name}/{data}/"
    feature_extraction("train", train_data, model, f_output_folder)
    feature_extraction("validation", validation_data, model, f_output_folder)
