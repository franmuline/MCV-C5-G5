import argparse as ap
from datasets import load_dataset
from feature_extraction import feature_extraction
from models import ResNet50

PATH_TO_DATA = "../"


def main():
    parser = ap.ArgumentParser(description="C5 - Week 3")
    parser.add_argument("--action", type=str, default="feature_extraction",
                        help="Action to perform, e.g. 'feature_extraction'")
    parser.add_argument("--model", type=str, default="ResNet50", help="Model to use, e.g. 'ResNet50'")
    parser.add_argument("--dataset", type=str, default="MIT_split", help="Dataset to use, e.g. 'MIT_split' or 'COCO'")
    parser.add_argument("--output_folder", type=str, default="./output", help="Output folder for the results")

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
        tag = model_name + "/" + data
        feature_extraction(tag + "_train", train_data, model, output_folder)
        feature_extraction(tag + "_validation", validation_data, model, output_folder)


if __name__ == "__main__":
    main()
