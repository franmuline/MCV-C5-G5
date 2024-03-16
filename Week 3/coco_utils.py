import json

def read_json_data(json_path: str):
    """Get json data from a given path."""
    with open(json_path, "r") as file:
        data = json.load(file)
    return data

# COCO classes
def get_categories(json_data: str):
    """
    Get categories from json data and return a dictionary
    with the category id as key and the category name as value.
    """
    categories = dict()
    for set in json_data["categories"]:
        id = set["id"]
        name = set["name"]
        supercategory = set["supercategory"]
        
        categories[id] = {"name": name, "supercategory": supercategory}

    return categories

def get_category_name(categories: list, category_id: int):
    """
    Get category name from a dict of categories.
    """
    return categories[0]["name"]

def get_category_id(categories: list, category_name: str):
    """
    Get category id from a dict of categories.
    """
    for category in categories:
        if category["name"] == category_name:
            return category["id"]
        
    return -1

def get_image_info(images_info: list, image_id: int):
    """
    Get image information from a list of dictionaries.
    """
    for image in images_info:
        if image["id"] == image_id:
            return image
    return None

def get_image_annotations(annotations: list, image_id: int):
    """
    Get annotations from a list of dictionaries.
    """
    anns = []
    for ann in annotations:
        if ann["image_id"] == image_id:
            anns.append(ann)
    return anns

def join_data(images_info: list, annotations: list):
    """
    Join information from images and annotations.
    """
    data = dict()
    for image in images_info:
        image_id = image["id"]
        anns = get_image_annotations(annotations, image_id)
        data[image_id] = image.copy()  # Make a copy of the image dictionary
        data[image_id]["annotations"] = anns
    return data

def get_image_objects(data: dict):
    image_objects = dict()
    for key, value in data.items():
        for image_id in value:
            if image_id not in image_objects:
                image_objects[image_id] = []
            image_objects[image_id].append(int(key))
    return image_objects

####################
####################
##### EXAMPLE ######
####################
####################

if __name__ == "__main__":
    dataset_path = "../COCO/"

    # Ground truth data
    mcv_json_data = read_json_data(dataset_path + "mcv_image_retrieval_annotations.json")
    train_gt = mcv_json_data["train"]  # Key: class_id, Value: list of image_ids
    val_gt = mcv_json_data["val"]      # Key: class_id, Value: list of image_ids

    # Load COCO data
    instances_train = read_json_data(dataset_path + "instances_train2014.json")
    instances_val = read_json_data(dataset_path + "instances_val2014.json")

    # Get COCO data
    # Dictionary -> id: [name, supercategory]
    coco_classes = get_categories(instances_train)

    # List of dictionaries:
    # keys -> [license, file_name, coco_url, height, width, date_captured, id]
    train_images_info = instances_train["images"]
    val_images_info = instances_val["images"]
    
    # List of dictionaries:
    # keys -> [segmentation, area, iscrowd, image_id, bbox, category_id, id]
    train_annotations = instances_train["annotations"]
    val_annotations = instances_val["annotations"]
    
    # Join information
    train_data = join_data(train_images_info, train_annotations)
    val_data = join_data(val_images_info, val_annotations)
