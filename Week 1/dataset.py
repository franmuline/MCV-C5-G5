import os
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


class MITSplitDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: Directory containing the dataset
        :param transform: Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))

        self.image_paths = []
        self.class_labels = []

        for i, cls in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls)
            for img in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img)
                self.image_paths.append(img_path)
                self.class_labels.append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.class_labels[idx]

        if self.transform:
            image = self.transform(image)

        image = TF.to_tensor(image)
        return image, label
