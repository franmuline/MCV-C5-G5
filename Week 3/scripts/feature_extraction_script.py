import os
import yaml

CONFIG_PATH = "../config/feature_extraction.yaml"
MODELS_PATH = "../models"

if __name__ == "__main__":
    list_of_models = os.listdir(MODELS_PATH)
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    for i, model in enumerate(sorted(list_of_models)):
        if i != 0:
            os.chdir("scripts/")
        # Write the model name to the config file and execute the main.py file with the config file as argument
        config["path_to_model"] = f"models/{model}"
        with open(CONFIG_PATH, "w") as file:
            yaml.dump(config, file)
        os.chdir("../")
        os.system(f"python main.py --action feature_extraction")