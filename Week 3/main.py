import cv2
import models
from torchvision import transforms


def main():
    model = models.ResNet50()

    # Load the image
    img = cv2.imread("../MIT_split/train/coast/arnat59.jpg")
    # Add a batch dimension
    img = transforms.ToTensor()(img).unsqueeze(0)
    # Perform inference
    out = model(img)

    print(out.shape)


if __name__ == "__main__":
    main()
