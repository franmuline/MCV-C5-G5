import yaml

from dataset import create_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog


def init(action, model):
    """Initialize the settings."""
    global dataset_config
    with open("config_files/dataset_config.yaml", "r") as file:
        dataset_config = yaml.safe_load(file)
        dataset_config = dataset_config["dataset_config"]
    global chosen_model
    chosen_model = model

    if action == "simple_inference":
        global model_name
        if model == "faster_rcnn":
            model_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        elif model == "mask_rcnn":
            model_name = "COCO-InstanceSegmentation/" + model + "_R_50_FPN_3x.yaml"

        global chosen_sequences
        with open("config_files/simple_inference/chosen_sequences.yaml", "r") as file:
            chosen_sequences = yaml.safe_load(file)
            chosen_sequences = chosen_sequences["sequences"]

