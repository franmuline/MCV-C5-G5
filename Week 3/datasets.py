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

class COCODataset(Dataset):
    def __init__(self, root, transform=None, train=False):
        self.N_CLASSES = 90
        self.root = root
        self.transform = transform

        anns = read_json_data(root[:root.rfind("/") + 1] + "mcv_image_retrieval_annotations.json")

        dir = root[root.rfind("/") + 1:]

        valid_anns = {}
        if "train2014" == dir:
            if not train:
                valid_anns = anns.get("database", {})
            else:
                valid_anns = anns.get("train", {})

        elif "val2014" == dir:
            valid_anns = anns.get("val", {})
            valid_anns.update({key: valid_anns.get(key, []) + anns["test"][key] for key in anns.get("test", {})})

        self.img_to_labels = get_image_objects(valid_anns)

        self.targets = [[] for _ in range(len(self.img_to_labels.keys()))]
        self.imgs = []

        for i, id in enumerate(self.img_to_labels.keys()):
            filename = f'COCO_{dir}_{id:012}.jpg'
            # Create an array of size N_CLASSES with 0s
            labels = np.zeros(self.N_CLASSES)
            for j in self.img_to_labels[id]:
                labels[j - 1] = 1
            self.targets[i] = labels
            self.imgs.append([root + "/" + filename, labels])

        self.labels_set = valid_anns.keys()

    def __getitem__(self, index):
        img, target = self.imgs[index]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)

        if img.shape[0] != 3:
            img = img.repeat(3, 1, 1)
        return img, target

    def __len__(self):
        return len(self.imgs)


class TripletCOCODataset(COCODataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform, True)

        self.label_to_indices = {}
        for i, x in enumerate(self.targets):
            # Get in the 90 one-hot encoded labels, the indices where the label is 1
            number_labels = np.where(x == 1)[0] + 1
            for label in number_labels:
                if label not in self.label_to_indices:
                    self.label_to_indices[label] = []
                self.label_to_indices[label].append(i)

        if "val" in root:
            self.test_triplets = self.get_triplets()

    def __getitem__(self, index):
        if "val" in self.root:
            img1, img2, img3 = self.test_triplets[index]
        else:
            img1, img2, img3 = self.get_triplet(index)
        img1 = Image.open(img1)
        img2 = Image.open(img2)
        img3 = Image.open(img3)
        # Check if they are grayscale and convert them to RGB
        if img1.mode != "RGB":
            img1 = img1.convert("RGB")
        if img2.mode != "RGB":
            img2 = img2.convert("RGB")
        if img3.mode != "RGB":
            img3 = img3.convert("RGB")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.imgs) if not "val" in self.root else len(self.test_triplets)

    def get_triplet(self, index):
        img1, labels1 = self.imgs[index]
        number_labels = np.where(labels1 == 1)[0] + 1
        img2 = img1
        # For the positive label, any image that shares a label with img1 is a valid positive label
        while img2 == img1:
            positive_label = np.random.choice(number_labels)
            img2, _ = self.imgs[np.random.choice(self.label_to_indices[positive_label])]

        # For the negative label, labels1 and labels3 cannot have any label in common
        while True:
            possible_negative_labels = list(set(self.label_to_indices.keys()) - set(number_labels))
            negative_label = np.random.choice(possible_negative_labels)
            chosen_negative_label = np.random.choice(self.label_to_indices[negative_label])
            img3, labels3 = self.imgs[chosen_negative_label]
            logical_array = np.logical_and(labels1, labels3)
            if not np.any(logical_array):
                break

        return img1, img2, img3

    def get_triplets(self):
        triplets = []
        for i in range(len(self.imgs)):
            img1, img2, img3 = self.get_triplet(i)
            triplets.append((img1, img2, img3))

        return triplets


def load_dataset(path: str, batch_size: int, shuffle: bool = False, type: str = "", n_samples: int = 0):
    """Load a dataset from a given path."""
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    if "COCO" not in path:
        if type == "siamese" and n_samples == 0:
            dataset = SiameseDataset(path, transform=transform)
        elif type == "triplet" and n_samples == 0:
            dataset = TripletDataset(path, transform=transform)
        else:
            dataset = ImageFolder(path, transform=transform)

            if n_samples > 0:
                # Get targets as tensor
                t_tensor = torch.tensor(dataset.targets)
                batch_sampler = BalancedBatchSampler(t_tensor, n_classes=len(dataset.classes), n_samples=n_samples)
                return DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        if type == "triplet" and n_samples == 0:
            dataset = TripletCOCODataset(path, transform=transform)
        else:
            dataset = COCODataset(path, transform=transform)
            if n_samples > 0:
                t_tensor = torch.tensor(dataset.targets)
                batch_sampler = BalancedBatchSampler(t_tensor, n_classes=dataset.N_CLASSES, n_samples=n_samples)
                return DataLoader(dataset, batch_sampler=batch_sampler)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
