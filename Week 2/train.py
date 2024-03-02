import os
import dataset
import yaml
import utils

from detectron2.engine import DefaultTrainer

def fine_tune(cfg, config, wandb):
    """Fine tune the given model with the given configuration
    :param cfg: The cfg to use
    :param config: Path to the configuration file to use
    :param wandb: Enable Weights & Biases logging
    """
    cfg.DATASETS.TRAIN = ("KITTI_train",)
    cfg.DATASETS.TEST = ("KITTI_validation",)
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300
    cfg.TEST.EVAL_PERIOD = 100
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if wandb:
        cfg = utils.setup_wandb(cfg)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()