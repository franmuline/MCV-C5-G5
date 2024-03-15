import os
import yaml
import argparse as ap
from datasets import load_dataset
from feature_extraction import perform_feature_extraction
from models import ResNet50, SiameseNet, TripletNet
from retrieval import retrieval
from visualization import visualize_UMAP
from losses import ContrastiveLoss, TripletsLoss
import torch.optim as optim
from train import *
from metrics import precision_at_k, recall_at_k, average_precision, plot_precision_recall_curve

PATH_TO_DATASET = "../"
PATH_TO_CONFIG = "./config/"


def main():
    parser = ap.ArgumentParser(description="C5 - Week 3")
    parser.add_argument("--action", type=str, default="evaluation",
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
        if data == "MIT_split":
            dataset = PATH_TO_DATA + "MIT_split"
        else:
            dataset = PATH_TO_DATA + "COCO"
        train_data = load_dataset(dataset + "/train", 16, True, metric)
        validation_data = load_dataset(dataset + "/test", 8, False, metric)

        # Set up network
        margin = 2.0
        embedding_net = ResNet50()

        model = None
        criterion = None
        if metric == "triplet":
            model = TripletNet(embedding_net).cuda()
            criterion = TripletsLoss(margin)
        elif metric == "siamese":
            model = SiameseNet(embedding_net).cuda()
            criterion = ContrastiveLoss(margin)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        n_epochs = 20
        log_interval = 20
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training
        for epoch in range(n_epochs):
            train_loss = train_epoch(model,optimizer,train_data,criterion,device,log_interval)

            val_loss = val_epoch(model,optimizer,validation_data,criterion,device,log_interval)

            print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss / len(train_data)} Val Loss: {val_loss / len(validation_data)}')
        # Save the model
        torch.save(model.state_dict(), metric+'_model.pth')

        print('Finished Training')


if __name__ == "__main__":
    main()
