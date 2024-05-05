import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn

from quantization_utils.bit_linear_custom import BitLinear

class AlexNetQuantized(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetQuantized, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.dropout = nn.Dropout()
        self.fc1 = BitLinear(256 * 6 * 6, 4096, bias = True)
        self.fc2 = BitLinear(4096, 4096, bias = True)
        self.fc3 = BitLinear(4096, num_classes, bias = True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.relu(self.fc1(x))

        x = self.dropout(x)
        x = self.relu(self.fc2(x))

        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):  # default for CIFAR-10
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096, bias = True)
        self.fc2 = nn.Linear(4096, 4096, bias = True)
        self.fc3 = nn.Linear(4096, num_classes, bias = True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.relu(self.fc1(x))

        x = self.dropout(x)
        x = self.relu(self.fc2(x))

        x = self.fc3(x)
        return x