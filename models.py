
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from fastai.layers import AdaptiveConcatPool2d
from delve import CheckLayerSat
from tqdm import tqdm, trange
from fastai.vision import learner

class SimpleFCNet(nn.Module):
    def __init__(self,in_channels: int, l1: int = 1024, l2: int = 512, l3: int = 256, n_classes: int = 10):
        super(SimpleFCNet, self).__init__()
        self.fc0 = nn.Linear(in_channels, l1)
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.fc3 = nn.Linear(l3, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3, l1: int = 8, l2: int = 16, l3: int = 32, n_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=l1, kernel_size=(3,3), stride=2)
        self.conv1 = nn.Conv2d(in_channels=l1, out_channels=l2, kernel_size=(3,3), stride=2)
        self.conv2 = nn.Conv2d(in_channels=l2, out_channels=l3, kernel_size=(3,3), stride=2)
        self.pool = AdaptiveConcatPool2d(1)
       # self.readout = nn.Flatten()
        self.out = nn.Linear(l3*2, n_classes)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x