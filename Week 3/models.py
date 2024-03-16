from torchvision import models
import torch.cuda as cuda
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.linear = nn.Linear(4096, embed_size)
        self.activation = nn.ReLU()
        self.device = "cuda" if cuda.is_available() else "cpu"

    def forward(self, x):
        x = x["pool"].flatten(start_dim=1)
        x = self.activation(x)
        x = self.linear(x)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2, progress=False)
        # Get rid of the last layer so the output will be the features
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        return self.model(x)


class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1").backbone
        self.model = nn.Sequential(*list(self.model.children())[:], EmbeddingLayer(embed_size=128))

    def forward(self, x):
        return self.model(x)

# CODE EXTRACTED FROM: https://github.com/adambielski/siamese-triplet/blob/master/networks.py
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
