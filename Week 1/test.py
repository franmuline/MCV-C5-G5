import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model import CustomNet
from dataset import MITSplitDataset
from sklearn.metrics import confusion_matrix, roc_curve, auc,ConfusionMatrixDisplay

if __name__ == '__main__':
    test_dir = '../MIT_split/test/'
    transform = transforms.Resize((64, 64))
    dataset_test = MITSplitDataset(test_dir, transform=transform)
    test = DataLoader(dataset_test, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = CustomNet()
    net.load_state_dict(torch.load('model_lr_0005_xavier_init.pth'))
    net.to(device)
    net.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test:
            inputs, labels = images.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    classes = dataset_test.classes

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.show()

    all_fpr = []
    all_tpr = []
    all_auc = []

    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_auc.append(auc(fpr, tpr))

    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        plt.plot(all_fpr[i], all_tpr[i], label=f'{classes[i]} (AUC = {all_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend()
    plt.grid()
    plt.show()
