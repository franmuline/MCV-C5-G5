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


def inf_on_image(img_path, predictor, cfg, coco_ids: bool = True):
    """Perform inference on a single image."""
    im = cv2.imread(img_path)

    outputs = predictor(im)
    if coco_ids:
        v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    else:
        v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(settings.dataset_config["name"] + "validation"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]


def simple_inference():
    """Perform inference with the two models on the selected sequences and store the results."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(settings.model_name))
    if settings.weights is not None:
        cfg.MODEL.WEIGHTS = settings.weights
        for d in ["train", "validation"]:
            DatasetCatalog.register(settings.dataset_config["name"] + d, lambda d=d: dataset.create_dataset(settings.dataset_config, d, False))
            MetadataCatalog.get(settings.dataset_config["name"] + d).set(thing_classes=settings.kitti_classes, stuff_classes=settings.kitti_classes)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(settings.kitti_classes)
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(settings.kitti_classes)
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(settings.model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    for sequence in settings.chosen_sequences:
        seq = settings.dataset_config["validation"]["directories"][sequence]
        sequence_path = (settings.dataset_config["path"] + settings.dataset_config["validation"]["prefix"] + seq)
        if settings.weights is not None:
            name_of_model = settings.weights.split("/")[-2]
            simple_inf_path = f"./simple_inference/{name_of_model}/{seq}"
        else:
            simple_inf_path = f"./simple_inference/{settings.chosen_model}/{seq}"
        if not os.path.exists(simple_inf_path):
            os.makedirs(simple_inf_path)
        for img in sorted(os.listdir(sequence_path)):
            img_path = sequence_path + "/" + img
            if settings.weights is not None:
                result = inf_on_image(img_path, predictor, cfg, False)
            else:
                result = inf_on_image(img_path, predictor, cfg, True)
            cv2.imwrite(f"{simple_inf_path}/{img}", result)


def inference():
    """Perform inference with the given model. The model can be a model name or a path to a model file."""
    coco_ids = settings.weights is None
    for d in ["train", "validation"]:
        DatasetCatalog.register(settings.dataset_config["name"] + d,
                                lambda d=d: dataset.create_dataset(settings.dataset_config, d, coco_ids))
        if coco_ids:
            MetadataCatalog.get(settings.dataset_config["name"] + d).set(thing_classes=settings.coco_classes,
                                                                         stuff_classes=settings.coco_classes)
        else:
            MetadataCatalog.get(settings.dataset_config["name"] + d).set(thing_classes=settings.kitti_classes,
                                                                         stuff_classes=settings.kitti_classes)

    cfg = get_cfg()
    cfg.defrost()
    cfg.merge_from_file(model_zoo.get_config_file(settings.model_name))
    if not coco_ids:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(settings.kitti_classes)
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(settings.kitti_classes)
    if coco_ids:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(settings.model_name)
        output_dir = f"./output/{settings.chosen_model}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        cfg.MODEL.WEIGHTS = settings.weights
        output_dir = settings.weights.split("/")[0:-1]
        output_dir = "/".join(output_dir)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATASETS.VAL = settings.dataset_config["name"] + "validation"
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(settings.dataset_config["name"] + "validation", output_dir=output_dir)
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


def inference_on_out_of_context():
    """Perform inference on the out of context dataset."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    list_of_images = os.listdir("../out_of_context")
    for img in list_of_images:
        im = cv2.imread(f"../out_of_context/{img}")
        outputs = predictor(im)
        # Make the font size larger
        v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get("coco_2017_val"), scale=1)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        if not os.path.exists("./out_of_context_results"):
            os.makedirs("./out_of_context_results")
        cv2.imwrite(f"./out_of_context_results/{img}", out.get_image()[:, :, ::-1])
