import json

from collections import defaultdict

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
            self.image_captions[image_id].append(ann["caption"])

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
