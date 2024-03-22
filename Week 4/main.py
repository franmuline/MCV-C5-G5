import argparse as ap
import fasttext
import fasttext.util

from collections import defaultdict
from utils import CaptionLoader, captions_to_word_vectors


PATH_TO_DATASET = "/ghome/mcv/datasets/C5/COCO/" # MCV server
# PATH_TO_DATASET = "../COCO/"                     # For local testing

PATH_TO_FASTTEXT_MODEL = "/ghome/mcv/C5/fasttext_wiki.en.bin" # MCV server
# PATH_TO_FASTTEXT_MODEL = "fasttext_wiki.en.bin"               # For local testing

def main():
    parser = ap.ArgumentParser(description="C5 - Week 4")
    parser.add_argument("--action", type=str, default="extract_text_features")

    args = parser.parse_args()
    action = args.action

    if action == "extract_text_features":
        train_loader = CaptionLoader(PATH_TO_DATASET + "captions_train2014.json")
        val_loader = CaptionLoader(PATH_TO_DATASET + "captions_val2014.json")

        # Load pre-trained FastText model
        print("Loading fastText model...")
        model = fasttext.load_model(PATH_TO_FASTTEXT_MODEL)

        model = fasttext.util.reduce_model(model, 100)  # Reducing vectors to dimension 100 (can be changed)

        # Trying with one word
        word_vector_dimension = model.get_word_vector("hello").shape
        print(word_vector_dimension)

        # Converting each caption into a word vector
        captions_vector = defaultdict(dict)
        for img_id, captions in train_loader.image_captions.items():
            captions_vector[img_id] = captions_to_word_vectors(captions, model)

    elif action == "image_to_text":
        print("Performing image to text retrieval...")

    elif action == "text_to_image":
        print("Performing text to image retrieval...")


if __name__ == "__main__":
    main()
