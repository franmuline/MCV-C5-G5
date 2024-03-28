import random
from PIL import Image

from torch.utils.data import Dataset
from .utils import read_json_data, SYMBOLS


class ImgCaptionsDataset(Dataset):
    def __init__(self, json_path, dataset_path, transform=None):
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
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.image_filename)

    def __getitem__(self, image_id):
        idxx = self.keys[image_id]
        image = Image.open(self.dataset_path + self.image_filename[idxx])
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        captions = self.get_image_captions(idxx)
        return image, captions

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
