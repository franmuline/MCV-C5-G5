import cv2
import os
import random
import yaml

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from pycocotools.mask import toBbox

COCO_classes = {
    1: 2,  # Car to Car
    2: 0  # Pedestrian to Person
}


def create_dataset(config: dict, dataset_type: str, coco_ids: bool = False):
    path = config["path"]
    cfg = config[dataset_type]
    labels_path = path + cfg["labels_path"]
    path = path + cfg["prefix"]
    directories = cfg["directories"]

    annotations = []
    for dir in sorted(os.listdir(path)):
        if dir in directories:
            gt_path = labels_path + dir + '.txt'
            with open(gt_path, 'r') as file:
                lines = file.read().splitlines()

            images_paths = sorted(os.listdir(path + dir))

            ann = []
            frame = 0
            for line in lines:
                splitted_line = line.split()
                actual_frame = int(splitted_line[0])

                if frame != actual_frame:
                    annotations.append({
                        "file_name": str(path + dir + '/' + images_paths[frame]),
                        "height": int(splitted_line[3]),
                        "width": int(splitted_line[4]),
                        "image_id": int(f"{dir}{frame:05}"),  # ESTO NOSE
                        "annotations": ann
                    })
                    frame = actual_frame
                    ann = []

                # object_id = int(splitted_line[1])
                class_id = int(splitted_line[2])
                obj_instance_id = int(splitted_line[1]) % 1000

                if (class_id != 10):
                    mask = {
                        "size": [int(splitted_line[3]), int(splitted_line[4])],
                        "counts": splitted_line[5]
                    }
                    bbox = toBbox(mask).tolist()
                    if coco_ids:
                        class_id = COCO_classes[class_id]
                    else:
                        class_id = class_id - 1
                    ann.append(
                        {
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "category_id": class_id,
                            "instance_id": obj_instance_id,  # NOSE SI HACE FALTA
                            "segmentation": mask,
                            "keypoints": [],
                            "iscrowd": 0
                        }
                    )
    return annotations
