import os
import yaml
import wandb
import argparse as ap
from datasets import load_dataset
from feature_extraction import perform_feature_extraction
from models import ResNet50, FasterRCNN, SiameseNet, TripletNet
from retrieval import retrieval
from visualization import visualize_UMAP, visualize_closest_images
from losses import ContrastiveLoss, TripletsLoss, OnlineTripletLoss, OnlineContrastiveLoss
from utils import get_pair_selector, get_triplet_selector
import torch.optim as optim
from train import *
from metrics import precision_at_k, recall_at_k, average_precision, plot_precision_recall_curve, save_results

PATH_TO_DATASET = "../"
PATH_TO_CONFIG = "./config/"
PATH_TO_ML_CONFIG = "./config/metric_learning/"


def main():
    parser = ap.ArgumentParser(description="C5 - Week 3")
    parser.add_argument("--action", type=str, default="metric_learning",
                        help="Action to perform, e.g. 'feature_extraction', 'retrieval', 'evaluation', 'visualization', 'metric_learning'. "
                             "To configure each action, please refer to the corresponding yaml file in the config folder.")
    parser.add_argument("--ml_config", type=str, default="metric_learning_coco.yaml",
                        help="Metric learning configuration file to use.")

    args = parser.parse_args()
    action = args.action
    ml_config = args.ml_config

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
        print(f"Performing retrieval with method {method} and parameters {params}")
        if "COCO" in folder:
            dim = 256
            coco = True
        else:
            dim = 2048
            coco = False
        indices, labels = retrieval(queries, features, method, params["metric"], dim, params["k"], coco)
        name = f"{folder}/{method}_{params['metric']}" if method == "knn" \
            else f"{folder}/{method}"
        if not os.path.exists(name):
            os.makedirs(name)

        np.save(name + "/retrieved_indices.npy", indices)
        np.save(name + "/retrieved_labels.npy", labels)

    elif action == "closest_images":
        with open(PATH_TO_CONFIG + "closest_images.yaml", "r") as file:
            config = yaml.safe_load(file)
        indices = config["indices_path"]
        data = config["data"]
        visualize_closest_images(PATH_TO_DATASET + data, indices)

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
        results_path = config["path_to_results"]
        gt = np.load(features_path)
        labels = np.load(retrieved_labels)
        if "COCO" not in retrieved_labels:
            binary_results = (labels == gt.reshape(-1, 1)).astype(int)
        else:
            # We have an (n_queries, k, 90) array, where at position (i, j) we have the 90-dimensional label vector
            # for the j-th retrieved image for the i-th query. To check if it is a true positive, we need to check if
            # the two images have at least one common label. That means doing a logical AND between the two 90-dimensional
            # vectors and checking if the result is different from the zero vector.
            current_labels = gt[:, -90:]
            current_labels = current_labels[:, np.newaxis, :]
            binary_results = np.any(np.logical_and(current_labels, labels), axis=2).astype(int)
        precisions = precision_at_k(binary_results)
        recalls = recall_at_k(binary_results)
        # Get path where the features are stored
        path = retrieved_labels.split("/")[:-1]
        model_name = path[-3]
        retrieval_method = path[-1]
        path = "/".join(path)

        print("Results for model", model_name, "and retrieval method", retrieval_method)

        mean_p_1 = np.mean(precisions[:, 0])
        mean_p_5 = np.mean(precisions[:, 4])
        print(f"Mean precision at 1: {mean_p_1}")
        print(f"Mean precision at 5: {mean_p_5}")

        # Compute the average precision
        avg_p = average_precision(binary_results)
        print(f"Mean average precision: {np.mean(avg_p)}")

        auc = plot_precision_recall_curve(precisions.mean(axis=0), recalls.mean(axis=0), path)
        print(f"AUC: {auc}")
        print("\n")
        save_results(model_name, retrieval_method, mean_p_1, mean_p_5, np.mean(avg_p), auc, results_path)

    elif action == "metric_learning":
        with open(PATH_TO_ML_CONFIG + ml_config, "r") as file:
            config = yaml.safe_load(file)
        dataset = PATH_TO_DATASET + config["data"]
        metric = config["metric"]
        loss = config["loss"]
        loss_params = config["loss_params"]
        batch_size = config["batch_size"]

        if "COCO" in dataset:
            train_dataset = dataset + "/train2014"
            val_dataset = dataset + "/val2014"
        else:
            train_dataset = dataset + "/train"
            val_dataset = dataset + "/test"

        if loss == "offline":
            train_data = load_dataset(train_dataset, batch_size, True, metric)
            validation_data = load_dataset(val_dataset, 8, False, metric)
        elif loss == "online":
            train_data = load_dataset(train_dataset, batch_size, True, metric, n_samples=1)
            validation_data = load_dataset(val_dataset, 8, False, metric, n_samples=1)
        else:
            raise ValueError("Loss type not supported")

        if "COCO" in dataset:
            embedding_net = ResNet50()  # Here we used the Faster RCNN, but did not work properly after a few epochs (more detail in power point)
        else:
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

        if loss == "offline":
            model_full_name = f"{metric}_{loss}_{loss_params['margin']}"
        elif loss == "online":
            model_full_name = f"{metric}_{loss}_{loss_params['selector']}_{loss_params['margin']}"
        else:
            raise ValueError("Loss type not supported")
        # Init a project in wandb with the name "week3", with the team "group-09-c3"
        wandb.init(project="c5_week3", entity="group-09-c3", name="COCO_" + model_full_name)

        print(
            f"Training {metric} with {loss} loss and parameters {loss_params} on dataset {dataset} with batch size {batch_size}...")

        # Training
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, optimizer, train_data, criterion, device, log_interval)

            val_loss = val_epoch(model, optimizer, validation_data, criterion, device, log_interval)

            print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss / len(train_data)} Val Loss: {val_loss / len(validation_data)}')
            # Log the train and validation loss in wandb
            wandb.log({"train_loss": train_loss / len(train_data), "val_loss": val_loss / len(validation_data)})

        # Save the model
        torch.save(model, "./models/" + model_full_name + ".pth")
        print('Finished Training')


if __name__ == "__main__":
    main()
