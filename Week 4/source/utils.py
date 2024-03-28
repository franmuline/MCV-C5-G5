import json
import torch
import random
from PIL import Image
import numpy as np
import yaml

from collections import defaultdict
from torch.utils.data import Dataset


PATH_TO_DATASET_TRAIN = "/ghome/mcv/datasets/C5/COCO/train2014/"  # For server

SYMBOLS = [".", ",", "!", "?", "'s", "'ll", "'re", "'m", "'ve", "'d",
            "(", ")", "[", "]", "{", "}", "<", ">", ":", ";", "-", "_",
            "=", "+", "*", "/", "\\", "|", "@", "#", "$", "%", "^", "&",
            "\"", "'", "—", "–", "\n", "\t", "\r", "\x0b", "\x0c"]


def read_json_data(json_path: str):
    """Get json data from a given path."""
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def read_yaml_data(yaml_path: str):
    """Get yaml data from a given path."""
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def img_caption_collate_fn(batch):
    """
    Custom collate_fn for the DataLoader. Images will be loaded as tensors and captions as lists of strings.
    """
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    # Transform images into a single tensor
    images = torch.stack(images)

    return images, captions


def captions_to_word_vectors(captions: list, model):
    """
    Captions: All captions for one image
    Convert a list of captions into a single vector.
    """
    captions_vector = []
    for caption in captions:
        words = caption.lower().split()
        caption_embedding = []
        for word in words:
            if word not in model.words():
                # FastText uses subwords to handle out-of-vocabulary words
                caption_embedding.append('<UNK>')
            else:
                caption_embedding.append(model.get_word_vector(word))

        if len(caption_embedding) != 0:
            combined_embedding = np.mean(caption_embedding, axis=0)
        else:
            combined_embedding = np.zeros(model.get_dimension())

        captions_vector.append(combined_embedding)

    final_caption_vector = np.mean(captions_vector, axis=0)  # Single vector mixing all captions through the mean

    return final_caption_vector
