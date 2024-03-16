import os
import numpy as np
import torch

from coco_utils import get_image_objects, read_json_data
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils import BalancedBatchSampler


class SiameseDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.labels_set = set(self.targets)

        self.label_to_indices = {}
        for label in self.labels_set:
            label_indices = [i for i, x in enumerate(self.targets) if x == label]
            self.label_to_indices[label] = np.array(label_indices)

        if "test" in self.root:
            self.test_pairs = self.get_negative_pairs() + self.get_positive_pairs()

    def __getitem__(self, index):

        if "test" in self.root:
            img1, img2, target = self.test_pairs[index]
            img1 = Image.open(img1)
            img2 = Image.open(img2)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return (img1, img2), target

        else:
            img1, label1 = self.imgs[index]
            target = np.random.randint(0, 2)
            if target == 1:
                siamese_index = index
                while siamese_index == index:  # keep looping till a different index is found
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2, label2 = self.imgs[siamese_index]
            img1 = Image.open(img1)
            img2 = Image.open(img2)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return (img1, img2), target

    def __len__(self):
        return len(self.imgs) if not "test" in self.root else len(self.test_pairs)

    def get_negative_pairs(self):
        negative_pairs = []
        for i in range(0, len(self.imgs), 2):
            img1, label1 = self.imgs[i]
            negative_index = np.random.choice(self.label_to_indices[np.random.choice(list(self.labels_set - set([label1])))])
            img2, _ = self.imgs[negative_index]
            negative_pairs.append((img1, img2, 0))

        return negative_pairs

    def get_positive_pairs(self):
        positive_pairs = []
        for i in range(1, len(self.imgs), 2):
            img1, label1 = self.imgs[i]
            positive_index = np.random.choice(self.label_to_indices[label1])
            img2, _ = self.imgs[positive_index]
            positive_pairs.append((img1, img2, 1))

        return positive_pairs


class TripletDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.labels_set = set(self.targets)

        self.label_to_indices = {}
        for label in self.labels_set:
            label_indices = [i for i, x in enumerate(self.targets) if x == label]
            self.label_to_indices[label] = np.array(label_indices)

        if "test" in self.root:
            self.test_triplets = self.get_triplets()

    def __getitem__(self, index):

        if "test" in self.root:
            img1, img2, img3 = self.test_triplets[index]
            img1 = Image.open(img1)
            img2 = Image.open(img2)
            img3 = Image.open(img3)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
            return (img1, img2, img3), []

        else:
            img1, label1 = self.imgs[index]

            positive_index = index
            while positive_index == index:  # keep looping till a different index is found
                positive_index = np.random.choice(self.label_to_indices[label1])

            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2, _ = self.imgs[positive_index]
            img3, _ = self.imgs[negative_index]

            img1 = Image.open(img1)
            img2 = Image.open(img2)
            img3 = Image.open(img3)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
            return (img1, img2, img3), []

    def __len__(self):
        return len(self.imgs) if not "test" in self.root else len(self.test_triplets)

    def get_triplets(self):
        triplets = []
        for i in range(len(self.imgs)):
            img1, label1 = self.imgs[i]

            positive_label = label1
            img2, _ = self.imgs[np.random.choice(self.label_to_indices[positive_label])]  # same label
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            img3, _ = self.imgs[np.random.choice(self.label_to_indices[negative_label])]  # different label

            triplets.append((img1, img2, img3))

        return triplets


class TripletCOCODataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        if "train2014" in root:
            instances = read_json_data(root[:root.rfind("/") + 1] + "instances_train2014.json")
        elif "val2014" in root:
            instances = read_json_data(root[:root.rfind("/") + 1] + "instances_val2014.json")

        categories = instances["categories"]
        self.labels_set = [name for _, name, _ in categories]

        anns = read_json_data(root[:root.rfind("/") + 1] + "mcv_image_retrieval_annotations.json")
        self.label_to_indices = anns["train"] if "train2014" in root else anns["val"]
        self.img_to_labels = get_image_objects(self.label_to_indices)

        self.imgs = []
        images_info = instances["images"]
        for img in os.listdir(root):
            for image in images_info:
                if image["file_name"] == img:
                    id = image["id"]
                    labels = []
                    if id in self.img_to_labels:
                        labels = self.img_to_labels[id]
                    self.imgs.append([root + "/" + img, labels])
                    break

        if "val" in root:
            self.test_triplets = self.get_triplets()

    def __getitem__(self, index):
        if "test" in self.root:
            img1, img2, img3 = self.test_triplets[index]
            img1 = Image.open(img1)
            img2 = Image.open(img2)
            img3 = Image.open(img3)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
            return (img1, img2, img3), []

        else:
            img1, labels1 = self.imgs[index]

            positive_index = index
            while positive_index == index:  # keep looping till a different index is found
                positive_index = np.random.choice(self.label_to_indices[np.random.choice(labels1)])

            negative_label = np.random.choice(list(set(self.labels_set) - set(labels1)))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2, _ = self.imgs[positive_index]
            img3, _ = self.imgs[negative_index]

            img1 = Image.open(img1)
            img2 = Image.open(img2)
            img3 = Image.open(img3)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
            return (img1, img2, img3), []

    def __len__(self):
        return len(self.imgs) if not "test" in self.root else len(self.test_triplets)

    def get_triplets(self):
        triplets = []
        for i in range(len(self.imgs)):
            img1, labels1 = self.imgs[i]

            img2 = img1
            while img2 == img1:
                positive_label = np.random.choice(labels1)
                img2, _ = self.imgs[np.random.choice(self.label_to_indices[positive_label])]


            negative_label = np.random.choice(list(set(self.labels_set) - set(labels1)))    # different label
            img3, _ = self.imgs[np.random.choice(self.label_to_indices[negative_label])]  

            triplets.append((img1, img2, img3))

        return triplets

def load_dataset(path: str, batch_size: int, shuffle: bool = False, type: str = "", n_samples: int = 0):
    """Load a dataset from a given path."""
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    if type == "siamese" and n_samples == 0:
        dataset = SiameseDataset(path, transform=transform)
    elif "COCO" in path:
        dataset = TripletCOCODataset(path, transform=transform)
    elif type == "triplet" and n_samples == 0:
        dataset = TripletDataset(path, transform=transform)
    else:
        dataset = ImageFolder(path, transform=transform)

        if n_samples > 0:
            # Get targets as tensor
            t_tensor = torch.tensor(dataset.targets)
            batch_sampler = BalancedBatchSampler(t_tensor, n_classes=len(dataset.classes), n_samples=n_samples)
            return DataLoader(dataset, batch_sampler=batch_sampler)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
