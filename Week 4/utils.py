import json

from collections import defaultdict

SYMBOLS = [".", ",", "!", "?", "'s", "'ll", "'re", "'m", "'ve", "'d",
            "(", ")", "[", "]", "{", "}", "<", ">", ":", ";", "-", "_",
            "=", "+", "*", "/", "\\", "|", "@", "#", "$", "%", "^", "&",
            "\"", "'", "—", "–"]


def read_json_data(json_path: str):
    """Get json data from a given path."""
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


class CaptionLoader:
    def __init__(self, json_path):
        data = read_json_data(json_path)

        self.image_filename = defaultdict(dict)
        self.image_captions = defaultdict(dict)
        for image in data["images"]:
            self.image_filename[image["id"]] = image["file_name"]
            self.image_captions[image["id"]] = [] # Initialize the captions

        for ann in data["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            for symbol in SYMBOLS:
                caption = caption.replace(symbol, " ")
            self.image_captions[image_id].append(caption)

    def get_image_filename(self, image_id: int):
        """
        Get image information.
        """
        return self.image_filename[image_id]
    
    def get_image_annotations(self, image_id: int):
        """
        Get annotations of a given image.
        """
        return self.image_captions[image_id]   


def captions_to_word_vectors(captions: list, model):
    """
    Convert a list of captions into a list of word vectors.
    """
    captions_vector = []
    for caption in captions:
        words = caption.lower().split()
        caption_embedding = []
        for word in words:
            if word not in model.words:
                # TODO: Handle unknown words -- Replace with Unknown Token
                caption_embedding.append('<UNK>') # FastText utilizes subwords to handle out-of-vocabulary words
                # print(f"Processing caption: {caption}")   # Debugging
                print(f"ERROR -- Word '{word}' not found in the model")   # Debugging
            else:
                caption_embedding.append(model.get_word_vector(word))
        captions_vector.append(caption_embedding)
    return captions_vector
