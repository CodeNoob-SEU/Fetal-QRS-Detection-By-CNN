from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 5))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 5))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 5))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 5))
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(4, 1))
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1, 5))
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1, 5))
        self.fc1 = nn.Linear(9728, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        x = x.view(-1, 9728)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

