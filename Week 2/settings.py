import yaml

from dataset import create_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog


def init():
    global dataset_config

    with open("./dataset_config.yaml", "r") as file:
        data = yaml.safe_load(file)

    dataset_config = data["dataset_config"]

    global phases
    phases = ["train", "validation"]

    global classes
    classes = ["car", "pedestrian"]

    global kitti
    kitti = "KITTI_"

    for d in phases:
        DatasetCatalog.register(kitti + d, lambda d=d: create_dataset(dataset_config, d))
        MetadataCatalog.get(kitti + d).set(thing_classes=classes)
