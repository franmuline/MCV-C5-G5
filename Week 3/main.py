import os
import argparse as ap
import numpy as np
from datasets import load_dataset
from feature_extraction import feature_extraction
from models import ResNet50, SiameseNet, TripletNet
from retrieval import retrieval
from losses import ContrastiveLoss, TripletsLoss
import torch.optim as optim
import torch
from train import *
from metrics import precision_at_k, recall_at_k, average_precision, plot_precision_recall_curve

PATH_TO_DATA = "../"


def main():
    parser = ap.ArgumentParser(description="C5 - Week 3")
    parser.add_argument("--action", type=str, default="retrieval",
                        help="Action to perform, e.g. 'feature_extraction', 'retrieval', 'metric_learning'")
    parser.add_argument("--model", type=str, default="ResNet50", help="Model to use, e.g. 'ResNet50'")
    parser.add_argument("--metric", type=str, default="siamese", help="Metric learning method to use: 'triplet', 'siamese'",
                        choices=["triplet", "siamese"])
    parser.add_argument("--dataset", type=str, default="MIT_split", help="Dataset to use, e.g. 'MIT_split' or 'COCO'")
    parser.add_argument("--output_folder", type=str, default="./output", help="Output folder for the results")
    parser.add_argument("--retrieval_method", type=str, default="knn",
                        help="Retrieval method to use, e.g. 'knn', 'faiss'")
    parser.add_argument("--retrieval_metric", type=str, default="cosine",
                        help="Retrieval metric to use, e.g. 'cosine', 'euclidean', 'manhattan'")
    parser.add_argument("--retrieval_k", type=int, default=None, help="Number of nearest neighbors to retrieve")

    args = parser.parse_args()
    action = args.action
    metric = args.metric
    model_name = args.model
    data = args.dataset
    output_folder = args.output_folder

    if action == "feature_extraction":
        if data == "MIT_split":
            dataset = PATH_TO_DATA + "MIT_split"
        else:
            dataset = PATH_TO_DATA + "COCO"
        train_data = load_dataset(dataset + "/train", 32, False)
        validation_data = load_dataset(dataset + "/test", 32, False)

        # Load the model
        if model_name == "ResNet50":
            model = ResNet50()
        else:
            # TODO: Add other models when trained Siamese and Triplet networks
            raise ValueError(f"Model {args.model} not available")
        f_output_folder = f"{output_folder}/{model_name}"
        feature_extraction(data + "_train", train_data, model, f_output_folder)
        feature_extraction(data + "_validation", validation_data, model, f_output_folder)

    elif action == "retrieval":
        folder = f"{output_folder}/{model_name}"
        features = np.load(f"{folder}/{data}_train_features_and_labels.npy")
        queries = np.load(f"{folder}/{data}_validation_features_and_labels.npy")
        indices, labels = retrieval(queries, features, args.retrieval_method, args.retrieval_metric, args.retrieval_k)
        name = f"{folder}/{data}_{args.retrieval_method}_{args.retrieval_metric}" if args.retrieval_method == "knn" \
            else f"{folder}/{data}_{args.retrieval_method}"
        if not os.path.exists(name):
            os.makedirs(name)

        np.save(name + "/retrieved_indices.npy", indices)
        np.save(name + "/retrieved_labels.npy", labels)

    elif action == "evaluation":
        # Get gt labels from validation set
        gt = np.load(f"{output_folder}/{model_name}/{data}_validation_features_and_labels.npy")[:, -1]
        # Load the retrieval labels
        if args.retrieval_method == "knn":
            labels = np.load(f"{output_folder}/{model_name}/{data}_{args.retrieval_method}_{args.retrieval_metric}/retrieved_labels.npy")
        else:
            labels = np.load(f"{output_folder}/{model_name}/{data}_{args.retrieval_method}/retrieved_labels.npy")

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
        train_data = load_dataset(dataset + "/train", 32, False, metric)
        validation_data = load_dataset(dataset + "/test", 32, False, metric)

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
        log_interval = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training
        for epoch in range(n_epochs):
            train_loss, train_acc = train_epoch(model,optimizer,train_data,criterion,device,log_interval)

            val_loss, val_acc = val_epoch(model,optimizer,validation_data,criterion,device,log_interval)

            print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss / len(train_data)}, '
              f'Train Accuracy: {train_acc}, Val Loss: {val_loss / len(validation_data)}, '
              f'Val Accuracy: {val_acc}')
        # Save the model
        torch.save(model.state_dict(), metric+'_model.pth')

        print('Finished Training')



if __name__ == "__main__":
    main()
