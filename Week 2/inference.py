import settings
import dataset
import cv2
import os
import sys

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog


def inf_on_image(img_path, predictor, cfg):
    """Perform inference on a single image."""
    im = cv2.imread(img_path)

    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]


def simple_inference():
    """Perform inference with the two models on the selected sequences and store the results."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(settings.model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(settings.model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    for sequence in settings.chosen_sequences:
        seq = settings.dataset_config["validation"]["directories"][sequence]
        sequence_path = (settings.dataset_config["path"] + settings.dataset_config["validation"]["prefix"] + seq)
        for img in sorted(os.listdir(sequence_path)):
            img_path = sequence_path + "/" + img
            result = inf_on_image(img_path, predictor, cfg)

            if not os.path.exists(f"./simple_inference/{settings.chosen_model}/{seq}"):
                os.makedirs(f"./simple_inference/{settings.chosen_model}/{seq}")
            cv2.imwrite(f"./simple_inference/{settings.chosen_model}/{seq}/{img}", result)


def inference(weights: str, coco_ids: bool = True):
    """Perform inference with the given model. The model can be a model name or a path to a model file."""
    for d in ["train", "validation"]:
        DatasetCatalog.register(settings.dataset_config["name"] + d, lambda d=d: dataset.create_dataset(settings.dataset_config, d, coco_ids))
        if coco_ids:
            MetadataCatalog.get(settings.dataset_config["name"] + d).set(thing_classes=settings.coco_classes, stuff_classes=settings.coco_classes)
        else:
            return  # TODO: Complete when no COCO ids are used

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(settings.model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(settings.model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATASETS.VAL = settings.dataset_config["name"] + "validation"
    predictor = DefaultPredictor(cfg)

    output_dir = f"./output/{settings.chosen_model}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    evaluator = COCOEvaluator(settings.dataset_config["name"] + "validation",  output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, settings.dataset_config["name"] + "validation")
    output_file = f"{output_dir}/output.txt"
    with open(output_file, "w") as file:
        # Redirect stdout to the file
        sys.stdout = file
        text = inference_on_dataset(predictor.model, val_loader, evaluator)
        print(text)
        # Reset stdout
        sys.stdout = sys.__stdout__
        file.close()
