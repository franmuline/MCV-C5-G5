import torch
import torch.nn as nn
import numpy as np
import fasttext
import fasttext.util
from transformers import BertModel, BertTokenizer
from concurrent.futures import ThreadPoolExecutor


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
    def __init__(self, model_path, embed_size, aggregation="mean"):
        super(FastText, self).__init__()
        self.model = fasttext.load_model(model_path)
        self.encoder = TextEncoder(300, embed_size)
        self.aggregation = aggregation

    def process_captions(self, captions):
        if self.aggregation == "mean":
            caption_embedding = []
            for caption in captions:
                words = caption.lower()
                caption_embedding.append(self.model.get_sentence_vector(words))
            caption_embedding = np.mean(caption_embedding, axis=0)
            return caption_embedding
        elif self.aggregation == "concat":
            concat_caption = ""
            for caption in captions:
                concat_caption += caption.lower() + " "
            return self.model.get_sentence_vector(concat_caption)
        else:
            raise ValueError("Aggregation method not supported")

    def forward(self, x):
        # X is a list of lists of captions (one list of captions per image). Each caption is a string of words.
        # We need to convert each caption into a single vector.
        # We will use the FastText model to get the sentence embeddings for each caption.
        with ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(self.process_captions, x))
        embeddings = np.array(embeddings)
        embeddings = self.encoder(embeddings)
        return embeddings

    def words(self):
        return self.model.words

    def get_word_vector(self, word):
        return self.model.get_word_vector(word)


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
