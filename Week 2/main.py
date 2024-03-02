import os
import dataset
import yaml
import random
import cv2
import argparse
import settings
import utils

from inference import inference
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
    parser.add_argument("--action", type=str, default="inference", help="Action to perform, can be 'inference', 'fine_tune' or 'random_search'")
    parser.add_argument("--model", type=str, default="mask_rcnn", help="Model to use, can be 'mask_rcnn', 'faster_rcnn' or a path to a model file")
    parser.add_argument("--config", type=str, default=None, help="For 'fine_tune' or 'random_search' actions, path to the configuration file to use")
    parser.add_argument("--wandb", type=bool, default=False, help="Enable Weights & Biases logging")
    parser.add_argument("--weights", type=str, default=None, help="Path to the weights file to use for inference")

    args = parser.parse_args()

    if args.action not in ["inference", "fine_tune", "random_search"]:
        raise ValueError("Invalid action")

    if args.action in ["fine_tune", "random_search"] and args.config is None:
        raise ValueError("You must provide a configuration file for 'fine_tune' or 'random_search' actions")

    if args.action in ["fine_tune", "random_search"] and not os.path.exists(args.config):
        raise ValueError("The configuration file does not exist")

    if args.model not in ["mask_rcnn", "faster_rcnn"]:
        raise ValueError("Invalid model")

    settings.init()

    metadata = MetadataCatalog.get("KITTI_train")

    cfg = utils.build_model(args.model, args.weights)
    if args.action == "inference":
        predictor = inference(cfg)
    elif args.action == "fine_tune":
        fine_tune(cfg, args.config, args.wandb)
    # elif args.action == "random_search":
    #     random_search(args.model, args.config, args.wandb)

    # with open("./dataset_config.yaml", "r") as file:
    #     data = yaml.safe_load(file)
    #
    # dataset_config = data["dataset_config"]
    #
    # for d in ["train", "validation"]:
    #     DatasetCatalog.register("KITTI_" + d, lambda d=d: dataset.create_dataset(dataset_config, d))
    #     MetadataCatalog.get("KITTI_" + d).set(thing_classes=["car", "pedestrian"])
    # metadata = MetadataCatalog.get("KITTI_train")
    #
    #
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = ("KITTI_train",)
    # cfg.DATASETS.TEST = ()
    # cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    # cfg.SOLVER.STEPS = []        # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # cfg.INPUT.MASK_FORMAT = "bitmask"
    #
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()
    #
    # # Inference should use the config with parameters that are used in training
    # # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    # predictor = DefaultPredictor(cfg)
    #
    # evaluator = COCOEvaluator("KITTI_validation", output_dir="./output")
    # val_loader = build_detection_test_loader(cfg, "KITTI_validation")
    # print(inference_on_dataset(predictor.model, val_loader, evaluator))
    #
    dataset_dicts = dataset.create_dataset(settings.dataset_config, "validation")
    for d in dataset_dicts[0:3]:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata,
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("IMAGE", out.get_image()[:, :, ::-1])

        # Wait for a key press and then close the window
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()