import argparse as ap
import fasttext
import fasttext.util

from utils import CaptionLoader

PATH_TO_DATASET = "/ghome/mcv/datasets/C5/COCO/" # MCV server
# PATH_TO_DATASET = "../COCO/"                     # For local testing

def main():
    parser = ap.ArgumentParser(description="C5 - Week 4")
    parser.add_argument("--action", type=str, default="image_to_text")

    args = parser.parse_args()
    action = args.action

    if action == "extract_text_features":
        train_loader = CaptionLoader("../COCO/captions_train2014.json")
        val_loader = CaptionLoader("../COCO/captions_val2014.json")

        # Load pre-trained FastText model
        print("Loading fastText model...")
        fasttext.util.download_model('en', if_exists='ignore')  # Only downloads it first time
        model = fasttext.load_model('cc.en.300.bin')  # Pre-trained word vectors have dimension 300, tarda 4min aprox

        model = fasttext.util.reduce_model(model, 100)  # Reducing vectors to dimension 100 (can be changed)

        # Trying with one word
        word_vector_dimension = model.get_word_vector("hello").shape
        print(word_vector_dimension)

        # TODO: no me ha dado tiempo
        # convert all captures to lowercase (la primera siempre suele estar en mayusculas)
        # para cada capture sacar el vector con get-word_vector() como en el ejemplo de "hello"

    elif action == "image_to_text":
        print("Performing image to text retrieval...")

    elif action == "text_to_image":
        print("Performing text to image retrieval...")


if __name__ == "__main__":
    main()
