import argparse as ap
from source import FastText, ResNet50, ImgTextCrossModel

from torch.utils.data import DataLoader
from torchvision import transforms

from collections import defaultdict
from source.utils import captions_to_word_vectors, img_caption_collate_fn, read_yaml_data
from source import ImgCaptionsDataset

PATH_TO_DATASET = "/ghome/mcv/datasets/C5/COCO/"  # MCV server
# PATH_TO_DATASET = "../COCO/"                     # For local testing



PATH_TO_FASTTEXT_MODEL = "/ghome/mcv/C5/fasttext_wiki.en.bin"  # MCV server
# PATH_TO_FASTTEXT_MODEL = "fasttext_wiki.en.bin"               # For local testing


def main():
    parser = ap.ArgumentParser(description="C5 - Week 4")
    parser.add_argument("--action", type=str, default="train")
    parser.add_argument("--ml_config", type=str, default="config/metric_learning/fasttext.yaml")

    args = parser.parse_args()
    action = args.action

    if action == "extract_text_features":
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        train_loader = ImgCaptionsDataset(PATH_TO_DATASET + "captions_train2014.json", PATH_TO_DATASET + "train2014/", transform=transform)
        train_dataloader = DataLoader(train_loader, batch_size=32, shuffle=False, collate_fn=img_caption_collate_fn)

        val_loader = ImgCaptionsDataset(PATH_TO_DATASET + "captions_val2014.json", PATH_TO_DATASET + "val2014/", transform=transform)
        val_dataloader = DataLoader(val_loader,batch_size=32, shuffle=False, collate_fn=img_caption_collate_fn)

        model = FastText(PATH_TO_FASTTEXT_MODEL, embed_size=2048)

        for i, (img, caps) in enumerate(train_dataloader):
            # print(i)
            # print(img)
            # print(caps)
            # Get embeddings for each caption with the FastText model
            embeddings = model(caps)
            print(f"BATCH {i}")

        # Load pre-trained FastText model
        print("Loading fastText model...")
        model = FastText(PATH_TO_FASTTEXT_MODEL, embed_size=2048)
        # model = fasttext.util.reduce_model(model, 100)  # Reducing vectors to dimension 100 (can be changed)

        # Trying with one word
        word_vector_dimension = model("hello")
        print(word_vector_dimension)

        # Converting each caption into a word vector
        captions_vector = defaultdict(dict)
        for img_id, captions in train_loader.image_captions.items():
            captions_vector[img_id] = captions_to_word_vectors(captions, model)

    elif action == "image_to_text":
        print("Performing image to text retrieval...")

    elif action == "text_to_image":
        print("Performing text to image retrieval...")

    elif action == "train":
        print("Loading configuration...")
        config = read_yaml_data(args.ml_config)

        print("Loading image model...")
        if config["img_model"] == "ResNet50":
            img_model = ResNet50()
            print("ResNet50 model loaded")
        else:
            raise ValueError("Image model not supported")

        print("Loading text model...")
        txt_model_config = config["text_model"]
        if txt_model_config["model_name"] == "fasttext":
            path = txt_model_config["model_path"]
            embed_size = txt_model_config["embedding_size"]
            strat = txt_model_config["strategy"]
            text_model = FastText(path, embed_size, strat)
            print("FastText model loaded")
        else:
            raise ValueError("Text model not supported")

        print("Building full model...")
        model = ImgTextCrossModel(img_model, text_model)
        print("Model built")
        dataset_path = config["data_dir"]

        print("Loading dataset...")
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        train_loader = ImgCaptionsDataset(dataset_path + "captions_train2014.json", dataset_path + "train2014/", transform=transform)
        train_dataloader = DataLoader(train_loader, batch_size=32, shuffle=False, collate_fn=img_caption_collate_fn)

        val_loader = ImgCaptionsDataset(dataset_path + "captions_val2014.json", dataset_path + "val2014/", transform=transform)
        val_dataloader = DataLoader(val_loader, batch_size=32, shuffle=False, collate_fn=img_caption_collate_fn)

        print("Training model...")
        for i, (img, caps) in enumerate(train_dataloader):
            img_embedding, text_embedding = model(img, caps)
            print(f"BATCH {i}")
            print(img_embedding)
            print(text_embedding)
            break






if __name__ == "__main__":
    main()
