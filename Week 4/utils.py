import json
import random
from PIL import Image
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset


PATH_TO_DATASET_TRAIN = "/ghome/mcv/datasets/C5/COCO/train2014/"  # For server

SYMBOLS = [".", ",", "!", "?", "'s", "'ll", "'re", "'m", "'ve", "'d",
            "(", ")", "[", "]", "{", "}", "<", ">", ":", ";", "-", "_",
            "=", "+", "*", "/", "\\", "|", "@", "#", "$", "%", "^", "&",
            "\"", "'", "—", "–"]


def read_json_data(json_path: str):
    """Get json data from a given path."""
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


class CaptionLoader(Dataset):
    def __init__(self, json_path, transform=None):
        data = read_json_data(json_path)

        self.image_filename = {}
        self.image_captions = {}
        for image in data["images"]:
            self.image_filename[int(image["id"])] = image["file_name"]
            self.image_captions[int(image["id"])] = []  # Initialize the captions

        for ann in data["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            for symbol in SYMBOLS:
                caption = caption.replace(symbol, " ")
            self.image_captions[image_id].append(caption)

        self.keys = list(self.image_filename.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_filename)

    def __getitem__(self, image_id):
        idxx = self.keys[image_id]
        image = Image.open(PATH_TO_DATASET_TRAIN + self.image_filename[idxx])
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        print(image.size)
        text_positive = self.get_positive_caption(idxx)
        text_negative = self.get_negative_caption(idxx)
        return (image,), text_positive, text_negative

    def get_image_filename(self, image_id: int):
        """
        Get image information.
        """
        return self.image_filename[image_id]
    
    def get_image_captions(self, image_id: int):
        """
        Get captions of a given image.
        """
        return self.image_captions[image_id]

    def get_positive_caption(self, image_id: int):
        """
        Get a positive caption for a given image.
        """
        return random.choice(self.image_captions[image_id])

    def get_negative_caption(self, image_id: int):
        """
        Get a negative caption for a given image.
        """
        image_ids = list(self.image_captions.keys())
        image_ids.remove(image_id)
        negative_image_id = random.choice(image_ids)
        return random.choice(self.image_captions[negative_image_id])


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
