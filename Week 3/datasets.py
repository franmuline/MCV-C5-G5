from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms


def load_dataset(path: str, batch_size: int, shuffle: bool = False):
    """Load a dataset from a given path."""
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = ImageFolder(path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

