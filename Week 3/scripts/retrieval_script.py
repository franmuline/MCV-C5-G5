import os
import yaml

CONFIG_PATH = "../config/retrieval.yaml"
OUTPUT_PATH = "../output"

if __name__ == "__main__":
    # List the directories in the output folder (only the directories)
    folders = [f for f in os.listdir(OUTPUT_PATH) if os.path.isdir(os.path.join(OUTPUT_PATH, f))]
    # Sort the folders
    folders.sort()
    # Iterate over the folders
    retrieval_methods = ["faiss"]
    for folder in folders:
        # Iterate over the retrieval methods
        for method in retrieval_methods:
            # Write the model name to the config file and execute the main.py file with the config file as argument
            with open(CONFIG_PATH, "r") as file:
                config = yaml.safe_load(file)
            config["path_to_train_features"] = f"output/{folder}/MIT_split/train_features_and_labels.npy"
            config["path_to_val_features"] = f"output/{folder}/MIT_split/validation_features_and_labels.npy"
            config["method"] = method.split("_")[0]
            config["params"]["metric"] = method.split("_")[1] if method != "faiss" else "None"
            with open(CONFIG_PATH, "w") as file:
                yaml.dump(config, file)
            os.chdir("../")
            os.system(f"python main.py --action retrieval")
            os.chdir("scripts")
