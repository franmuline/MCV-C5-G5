import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2, progress=False)
        # Get rid of the last layer so the output will be the features
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        return self.model(x)
