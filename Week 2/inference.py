import settings
import dataset
import cv2
import os

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
    v = Visualizer(im[:, :, ::-1],
                     metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2
    )
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


def inference():
    """Perform inference with the given model. The model can be a model name or a path to a model file."""

    settings.init()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("KITTI_validation", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "KITTI_validation")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    metadata = MetadataCatalog.get("KITTI_train")

    dataset_dicts = dataset.create_dataset(settings.dataset_config, "validation")
    for d in dataset_dicts[0:3]:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("IMAGE", out.get_image()[:, :, ::-1])

        # Wait for a key press and then close the window
        cv2.waitKey(0)
    cv2.destroyAllWindows()

