import settings
import dataset
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


def inference(cfg):
    """Perform inference with the given model. The model can be a model name or a path to a model file.
    :param cfg: The cfg to use
    """
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("KITTI_validation", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "KITTI_validation")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    return predictor
