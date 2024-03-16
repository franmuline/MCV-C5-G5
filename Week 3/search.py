import os

CONFIG_PATH = "./config/metric_learning"

if __name__ == "__main__":
    files = os.listdir(CONFIG_PATH)
    for file in sorted(files):
        # Execute the main.py file with the config file as argument
        os.system(f"python main.py --action metric_learning --ml_config {file}")
