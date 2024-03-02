import os
import dataset
import yaml
import random
import cv2
import argparse
import settings
import utils

from inference import inference, simple_inference
from train import fine_tune
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog


def main():
    # Read arguments
    parser = argparse.ArgumentParser(description="C5 - Week 2")
    parser.add_argument("--action", type=str, default="simple_inference", help="Action to perform, can be 'simple_inference', 'eval_inference', 'fine_tune' or 'random_search'")
    parser.add_argument("--model", type=str, default="faster_rcnn", help="Model to use, can be 'mask_rcnn', 'faster_rcnn' or a path to a model file")
    parser.add_argument("--config", type=str, default=None, help="For 'fine_tune' or 'random_search' actions, path to the configuration file to use")
    parser.add_argument("--wandb", type=bool, default=False, help="Enable Weights & Biases logging")
    parser.add_argument("--weights", type=str, default=None, help="Path to the weights file to use for inference")

    args = parser.parse_args()

    settings.init(args.action, args.model)

    if args.action == "simple_inference":
        # Performs inference on the selected sequences and stores the result images
        simple_inference()


if __name__ == "__main__":
    main()