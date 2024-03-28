import torch.nn as nn


class ImgTextCrossModel(nn.Module):
    def __init__(self, img_encoder, text_encoder):
        super(ImgTextCrossModel, self).__init__()
        self.img_encoder = img_encoder
        self.text_encoder = text_encoder

    def forward(self, img, text):
        img_embedding = self.img_encoder(img)
        text_embedding = self.text_encoder(text)
        return img_embedding, text_embedding

    def img_forward(self, img):
        return self.img_encoder(img)

    def text_forward(self, text):
        return self.text_encoder(text)