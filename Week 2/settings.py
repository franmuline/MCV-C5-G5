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

    global model_name
    if model == "faster_rcnn":
        model_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    elif model == "mask_rcnn":
        model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    global coco_classes
    coco_classes = [""] * 81
    coco_classes[0] = "person"
    coco_classes[2] = "car"

    global kitti_classes
    kitti_classes = ["car", "pedestrian"]

    global weights
    with open("config_files/weights.yaml", "r") as file:
        weights = yaml.safe_load(file)
        weights = weights["weights"]
        if weights == "None":
            weights = None

    if action == "simple_inference":
        global chosen_sequences
        with open("config_files/chosen_sequences.yaml", "r") as file:
            chosen_sequences = yaml.safe_load(file)
            chosen_sequences = chosen_sequences["sequences"]


