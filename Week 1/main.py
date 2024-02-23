import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import MITSplitDataset
from model import CustomNet

if __name__ == '__main__':
    wandb.init(project='c5_week1')

    train_dir = '../MIT_small_train_1_augmented/train/'
    val_dir = '../MIT_small_train_1_augmented/test/'

    transform = transforms.Resize((64, 64))
    dataset_training = MITSplitDataset(train_dir, transform=transform)
    dataset_validation = MITSplitDataset(val_dir, transform=transform)

    training = DataLoader(dataset_training, batch_size=16, shuffle=True)
    validation = DataLoader(dataset_validation, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(net.parameters(), lr=0.0001)

    for epoch in range(50):
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in training:
            inputs, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total

        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in validation:
                inputs, labels = images.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        wandb.log({'epoch': epoch + 1,
                   'train_loss': train_loss / len(training),
                   'train_accuracy': train_accuracy,
                   'val_loss': val_loss / len(validation),
                   'val_accuracy': 100. * correct / total})

        print(f'Epoch {epoch + 1}/{50}, Train Loss: {train_loss / len(training)}, '
              f'Train Accuracy: {train_accuracy}, Val Loss: {val_loss / len(validation)}, '
              f'Val Accuracy: {100. * correct / total}')

    # Save the model
    torch.save(net.state_dict(), 'model.pth')

    print('Finished Training')
    wandb.finish()
