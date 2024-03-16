import os
import yaml
import argparse as ap
from datasets import load_dataset
from feature_extraction import perform_feature_extraction
from models import ResNet50, SiameseNet, TripletNet
from retrieval import retrieval
from visualization import visualize_UMAP
from losses import ContrastiveLoss, TripletsLoss, OnlineTripletLoss, OnlineContrastiveLoss
from utils import get_pair_selector, get_triplet_selector
import torch.optim as optim
from train import *
from metrics import precision_at_k, recall_at_k, average_precision, plot_precision_recall_curve

PATH_TO_DATASET = "../"
PATH_TO_CONFIG = "./config/"


def main():
    parser = ap.ArgumentParser(description="C5 - Week 3")
    parser.add_argument("--action", type=str, default="metric_learning",
                        help="Action to perform, e.g. 'feature_extraction', 'retrieval', 'evaluation', 'visualization', 'metric_learning'")
    parser.add_argument("--metric", type=str, default="triplet", help="Metric learning method to use: 'triplet', 'siamese'",
                        choices=["triplet", "siamese"])
    parser.add_argument("--dataset", type=str, default="MIT_split", help="Dataset to use, e.g. 'MIT_split' or 'COCO'")
    parser.add_argument("--output_folder", type=str, default="./output", help="Output folder for the results")

    args = parser.parse_args()
    action = args.action
    metric = args.metric
    data = args.dataset

    if action == "feature_extraction":
        # Read arguments from yaml file
        with open(PATH_TO_CONFIG + "feature_extraction.yaml", "r") as file:
            config = yaml.safe_load(file)
        dataset = config["data"]
        model_path = config["path_to_model"]
        output_path = config["output_path"]
        perform_feature_extraction(PATH_TO_DATASET, dataset, model_path, output_path)

    elif action == "retrieval":
        with open(PATH_TO_CONFIG + "retrieval.yaml", "r") as file:
            config = yaml.safe_load(file)
        features = np.load(config["path_to_train_features"])
        queries = np.load(config["path_to_val_features"])
        method = config["method"]
        params = config["params"]
        folder = config["path_to_train_features"].split("/")[:-1]
        folder = "/".join(folder)
        indices, labels = retrieval(queries, features, method, params["metric"], params["k"])
        name = f"{folder}/{method}_{params['metric']}" if method == "knn" \
            else f"{folder}/{method}"
        if not os.path.exists(name):
            os.makedirs(name)

        np.save(name + "/retrieved_indices.npy", indices)
        np.save(name + "/retrieved_labels.npy", labels)

    elif action == "visualization":
        with open(PATH_TO_CONFIG + "visualization.yaml", "r") as file:
            config = yaml.safe_load(file)
        features_path = config["path_to_features"]
        type = config["type"]
        if type == "UMAP":
            params = config["params"]
            n_components = params["n_components"]
            min_dist = params["min_dist"]
            n_neighbors = params["n_neighbors"]
            metric = params["metric"]
            visualize_UMAP(features_path, n_components, min_dist, n_neighbors, metric)
        else:
            raise ValueError("Visualization type not supported")

    elif action == "evaluation":
        with open(PATH_TO_CONFIG + "evaluation.yaml", "r") as file:
            config = yaml.safe_load(file)
        features_path = config["path_to_features"]
        retrieved_labels = config["path_to_labels"]
        gt = np.load(features_path)[:, -1]
        labels = np.load(retrieved_labels)
        binary_results = (labels == gt.reshape(-1, 1)).astype(int)
        precisions = precision_at_k(binary_results)
        recalls = recall_at_k(binary_results)

        print(f"Mean precision at 1: {np.mean(precisions[:, 0])}")
        print(f"Mean precision at 5: {np.mean(precisions[:, 4])}")

        # Compute the average precision
        avg_p = average_precision(binary_results)
        print(f"Mean average precision: {np.mean(avg_p)}")

        plot_precision_recall_curve(precisions.mean(axis=0), recalls.mean(axis=0))

    elif action == "metric_learning":
        with open(PATH_TO_CONFIG + "metric_learning.yaml", "r") as file:
            config = yaml.safe_load(file)
        dataset = PATH_TO_DATASET + config["data"]
        metric = config["metric"]
        loss = config["loss"]
        loss_params = config["loss_params"]
        batch_size = config["batch_size"]

        if loss == "offline":
            train_data = load_dataset(dataset + "/train", batch_size, True, metric)
            validation_data = load_dataset(dataset + "/test", 8, False, metric)
        elif loss == "online":
            train_data = load_dataset(dataset + "/train", batch_size, True, metric, n_samples=3)
            validation_data = load_dataset(dataset + "/test", 8, False, metric, n_samples=3)
        else:
            raise ValueError("Loss type not supported")

        embedding_net = ResNet50()

        if metric == "siamese":
            if loss == "offline":
                model = SiameseNet(embedding_net).cuda()
                criterion = ContrastiveLoss(loss_params["margin"])
            elif loss == "online":
                model = embedding_net.cuda()
                pair_selector = get_pair_selector(loss_params["selector"])
                criterion = OnlineContrastiveLoss(loss_params["margin"], pair_selector)
            else:
                raise ValueError("Loss type not supported")
        elif metric == "triplet":
            if loss == "offline":
                model = TripletNet(embedding_net).cuda()
                criterion = TripletsLoss(loss_params["margin"])
            elif loss == "online":
                model = embedding_net.cuda()
                triplet_selector = get_triplet_selector(loss_params["selector"], loss_params["margin"])
                criterion = OnlineTripletLoss(loss_params["margin"], triplet_selector)
            else:
                raise ValueError("Loss type not supported")
        else:
            raise ValueError("Metric type not supported")

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        n_epochs = config["n_epochs"]
        log_interval = config["log_interval"]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, optimizer, train_data, criterion, device, log_interval)

            val_loss = val_epoch(model, optimizer, validation_data, criterion, device, log_interval)

            print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss / len(train_data)} Val Loss: {val_loss / len(validation_data)}')
        # Save the model
        torch.save(model.state_dict(), metric+'_model.pth')

        print('Finished Training')


if __name__ == "__main__":
    main()
