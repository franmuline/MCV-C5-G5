import cv2
import os
import random
import yaml

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from pycocotools.mask import toBbox


def create_dataset(config: dict, dataset_type: str):
    path = config["path"]
    cfg = config[dataset_type]
    labels_path = path + cfg["labels_path"]
    path = path + cfg["prefix"]
    directories = cfg["directories"]

    annotations = []
    for dir in os.listdir(path):
        if dir in directories:
            gt_path = labels_path + dir +'.txt'
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
                        "width":  int(splitted_line[4]),
                        "image_id": int(f"{dir}{frame:05}"), # ESTO NOSE
                        "annotations": ann
                    })
                    frame = actual_frame
                    ann = []

                #object_id = int(splitted_line[1])
                class_id = int(splitted_line[2]) - 1
                obj_instance_id = int(splitted_line[1]) % 1000

                if (class_id != 10 - 1):
                    mask = {
                        "size": [int(splitted_line[3]), int(splitted_line[4])],
                        "counts": splitted_line[5]
                    }
                    bbox = toBbox(mask).tolist() 
                    ann.append(
                        {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": class_id,
                        "instance_id": obj_instance_id, # NOSE SI HACE FALTA
                        "segmentation": mask,
                        "keypoints": [],
                        "iscrowd": 0
                        }
                    )
    return annotations            


if __name__ == "__main__":

    with open("./dataset_config.yaml", "r") as file:
        data = yaml.safe_load(file)

    dataset_config = data["dataset_config"]
    create_dataset(dataset_config, "train")
    create_dataset(dataset_config, "validation")

    for d in ["train", "validation"]:
        DatasetCatalog.register("KITTI_" + d, lambda d=d: create_dataset(dataset_config, d))
        MetadataCatalog.get("KITTI_" + d).set(thing_classes=["car", "pedestrian"])
    metadata = MetadataCatalog.get("KITTI_train")


    # Print 3 random images
    dataset_dicts = create_dataset(dataset_config, "train")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("Image", out.get_image()[:, :, ::-1])

        # Wait for a key press and then close the window
        cv2.waitKey(0)
    cv2.destroyAllWindows()