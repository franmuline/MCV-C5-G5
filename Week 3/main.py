import argparse as ap
import numpy as np
from datasets import load_dataset
from feature_extraction import feature_extraction
from models import ResNet50
from retrieval import retrieval
from losses import ContrastiveLoss, TripletsLoss
from networks import EmbeddingNet, SiameseNet, TripletNet
import torch
import torch.optim as optim

PATH_TO_DATA = "../"


def main():
    parser = ap.ArgumentParser(description="C5 - Week 3")
    parser.add_argument("--action", type=str, default="retrieval",
                        help="Action to perform, e.g. 'feature_extraction', 'retrieval', 'siamese_network', "
                             "'triplet_network'")
    parser.add_argument("--model", type=str, default="ResNet50", help="Model to use, e.g. 'ResNet50'")
    parser.add_argument("--dataset", type=str, default="MIT_split", help="Dataset to use, e.g. 'MIT_split' or 'COCO'")
    parser.add_argument("--output_folder", type=str, default="./output", help="Output folder for the results")
    parser.add_argument("--retrieval_method", type=str, default="knn",
                        help="Retrieval method to use, e.g. 'knn', 'faiss'")
    parser.add_argument("--retrieval_metric", type=str, default="cosine",
                        help="Retrieval metric to use, e.g. 'cosine', 'euclidean'")
    parser.add_argument("--retrieval_k", type=int, default=None, help="Number of nearest neighbors to retrieve")

    args = parser.parse_args()
    action = args.action
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
        indices = retrieval(queries, features, args.retrieval_method, args.retrieval_metric, args.retrieval_k)
        name = f"{folder}/{data}_{args.retrieval_method}_{args.retrieval_metric}_retrieval_indices.npy" if args.retrieval_method == "knn" \
            else f"{folder}/{data}_{args.retrieval_method}_retrieval_indices.npy"
        np.save(name, indices)

    elif action == "siamese_network":
        if data == "MIT_split":
            dataset = PATH_TO_DATA + "MIT_split"
        else:
            dataset = PATH_TO_DATA + "COCO"
        train_data = load_dataset(dataset + "/train", 32, False)
        validation_data = load_dataset(dataset + "/test", 32, False)

        # Set up network
        margin = 2.0
        embedding_net = EmbeddingNet()
        model_siamese = SiameseNet(embedding_net).cuda()
        criterion = ContrastiveLoss(margin)
        optimizer = optim.Adam(model_siamese.parameters(), lr=1e-3)
        n_epochs = 20
        log_interval = 100

        # Training

    elif action == "triplet_network":
        if data == "MIT_split":
            dataset = PATH_TO_DATA + "MIT_split"
        else:
            dataset = PATH_TO_DATA + "COCO"
        train_data = load_dataset(dataset + "/train", 32, False)
        validation_data = load_dataset(dataset + "/test", 32, False)

        # Set up network
        margin = 2.0
        embedding_net = EmbeddingNet()
        model_triplet = TripletNet(embedding_net).cuda()
        criterion = TripletsLoss(margin)
        optimizer = optim.Adam(model_triplet.parameters(), lr=1e-3)
        n_epochs = 20
        log_interval = 100

        # Training



if __name__ == "__main__":
    main()
