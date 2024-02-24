import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        # convBlock(x2), 8 filters, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 8, 3)  # input channels, filters, kernel size
        self.conv2 = nn.Conv2d(8, 8, 3)
        # convBlock(x2), 16 filters, 3x3 kernels
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.conv4 = nn.Conv2d(16, 16, 3)
        # convBlock(x3), 32 filters, 3x3 kernels
        self.conv5 = nn.Conv2d(16, 32, 3)
        self.conv6 = nn.Conv2d(32, 32, 3)
        self.conv7 = nn.Conv2d(32, 32, 3)
        # fully-connected layer 128 units
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        # batchNormalization
        self.bn = nn.BatchNorm1d(128)
        # dropout
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 8)  # 8 classes

        # initialize weights
        init.xavier_normal_(self.conv1.weight)
        init.xavier_normal_(self.conv2.weight)
        init.xavier_normal_(self.conv3.weight)
        init.xavier_normal_(self.conv4.weight)
        init.xavier_normal_(self.conv5.weight)
        init.xavier_normal_(self.conv6.weight)
        init.xavier_normal_(self.conv7.weight)
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # maxPooling first block (2x2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        # maxPooling second block (2x2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        # maxPooling third block (2x2)
        x = F.max_pool2d(F.relu(self.conv7(x)), 2)
        # flatten
        x = torch.flatten(x, 1)
        # dense layer
        x = F.relu(self.fc1(x))
        # batchNormalization + relu activation
        x = F.relu(self.bn(x))
        # dropout 0.5
        x = self.dropout(x)
        # softmax
        x = F.softmax(F.relu(self.fc2(x)), 1)
        return x
