import argparse
import settings

from inference import inference, simple_inference, inference_on_out_of_context
from train import random_search


def main():
    # Read arguments
    parser = argparse.ArgumentParser(description="C5 - Week 2")
    parser.add_argument("--action", type=str, default="out_of_context", help="Action to perform, can be 'simple_inference', 'eval_inference', 'random_search' or 'out_of_context'")
    parser.add_argument("--model", type=str, default="mask_rcnn", help="Model to use, can be 'mask_rcnn' or 'faster_rcnn'")

    args = parser.parse_args()

    settings.init(args.action, args.model)

    if args.action == "simple_inference":
        # Performs inference on the selected sequences and stores the result images
        simple_inference()
    elif args.action == "eval_inference":
        # Performs inference with the given model and evaluates the results
        inference()
    elif args.action == "random_search":
        # Performs random search with the given configuration
        random_search(9)
    elif args.action == "out_of_context":
        inference_on_out_of_context()


if __name__ == "__main__":
    main()
