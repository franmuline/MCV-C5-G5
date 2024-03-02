from detectron2 import model_zoo
from detectron2.config import get_cfg


def build_model(model, weights):
    """Build the cfg for the given model and weights
    :param model: Model to use
    :param weights: Path to the weights file to use
    :return: The cfg
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{model}_R_50_FPN_3x.yaml"))
    if weights is not None:
        cfg.MODEL.WEIGHTS = weights
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{model}_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.TEST.EVAL_PERIOD = 100
    return cfg

def setup_wandb(cfg):
    """Setup Weights & Biases logging
    :param cfg: The cfg to use
    :return: The cfg with Weights & Biases logging enabled
    """
    cfg.WANDB.ENABLED = True
    cfg.WANDB.LOG_MODEL = True
    return cfg
