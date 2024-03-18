import os
import yaml

CONFIG_PATH = "../config/evaluation.yaml"
OUTPUT_PATH = "../output"

if __name__ == "__main__":
    print(os.listdir("./"))
    # List the directories in the output folder (only the directories)
    folders = [f for f in os.listdir(OUTPUT_PATH) if os.path.isdir(os.path.join(OUTPUT_PATH, f))]
    # Sort the folders
    folders.sort()
    # Iterate over the folders
    for folder in folders:
        # Write the model name to the config file and execute the main.py file with the config file as argument
        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)
        config["path_to_features"] = f"output/{folder}/MIT_split/validation_features_and_labels.npy"
        # Only directories in the MIT_split folder are retrieval methods
        methods = [m for m in os.listdir(OUTPUT_PATH + "/" + folder + "/MIT_split/")
                   if os.path.isdir(os.path.join(OUTPUT_PATH + "/" + folder + "/MIT_split/", m))]
        for method in sorted(methods):
            config["path_to_labels"] = f"output/{folder}/MIT_split/{method}/retrieved_labels.npy"
            with open(CONFIG_PATH, "w") as file:
                yaml.dump(config, file)
            os.chdir("../")
            os.system(f"python main.py --action evaluation")
            os.chdir("scripts")
