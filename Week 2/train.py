import os
import dataset
import settings
import wandb

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("./output", exist_ok=True)
            output_folder = "./output"
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def fine_tune():
    """Fine tune the given model with the given configuration"""
    # Put as name the model name and a random name
    wandb.init(project="c5-week2", config=settings.dataset_config, sync_tensorboard=True)
    config = wandb.config
    for d in ["train", "validation"]:
        DatasetCatalog.register(settings.dataset_config["name"] + d, lambda d=d: dataset.create_dataset(settings.dataset_config, d))
        MetadataCatalog.get(settings.dataset_config["name"] + d).set(thing_classes=settings.kitti_classes, stuff_classes=settings.kitti_classes)

    metadata = MetadataCatalog.get(settings.dataset_config["name"] + "validation")

    cfg = get_cfg()
    cfg.defrost()
    cfg.merge_from_file(model_zoo.get_config_file(settings.model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(settings.model_name)
    cfg.DATASETS.TRAIN = (settings.dataset_config["name"] + "train",)
    cfg.DATASETS.TEST = (settings.dataset_config["name"] + "validation",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = config.images_per_batch
    cfg.SOLVER.BASE_LR = config.learning_rate
    cfg.SOLVER.MAX_ITER = 5000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(settings.kitti_classes)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.OUTPUT_DIR = f"./output/fine_tuned_{settings.chosen_model}_{wandb.run.name}"
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.TEST.EVAL_PERIOD = 500

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def random_search(count=10):
    wandb.login()
    sweep_config = {
        'name': 'Hyperparameter tuning',
        'method': 'random',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'model': {
                'values': [settings.chosen_model]
            },
            'learning_rate': {
                'distribution': 'uniform',
                'min': 0.00001,
                'max': 0.0025
            },
            'batch_size_per_image': {
                'values': [32, 64, 128]
            },
            'images_per_batch': {
                'values': [2, 4, 8]
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="c5_week2")
    wandb.agent(sweep_id, function=fine_tune, count=count)
