import torch
import torch.nn as nn
import numpy as np
import fasttext
import fasttext.util
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):
    def __init__(self, input_size, embed_size):
        super(TextEncoder, self).__init__()
        # Add a layer to increase the dimension of the word vectors, to match the image features
        self.linear = nn.Linear(input_size, embed_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        # If the input is a 1D tensor, add a dimension to match the image features
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.activation(x)
        x = self.linear(x)
        x = x / x.pow(2).sum(dim=1, keepdim=True).sqrt()
        return x


class FastText(nn.Module):
    def __init__(self, model_path, embed_size):
        super(FastText, self).__init__()
        self.model = fasttext.load_model(model_path)
        self.encoder = TextEncoder(300, embed_size)

    def forward(self, x):
        x = self.model.get_word_vector(x)
        x = self.encoder(x)
        return x

    def words(self):
        return self.model.words


class Bert(nn.Module):
    def __init__(self, model_name, embed_size):
        super(Bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.encoder = TextEncoder(768, embed_size)

    def forward(self, x):
        x = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        x = self.model(**x).last_hidden_state.mean(dim=1)
        x = self.encoder(x)
        return x

    def words(self):
        return self.tokenizer.get_vocab().keys()
