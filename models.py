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
    def __init__(self,
                 in_channels: int,
                 l1: int = 1024,
                 l2: int = 512,
                 l3: int = 256,
                 n_classes: int = 10):
        super(SimpleFCNet, self).__init__()

        print('Setting up FCN with: l1', l1, 'l2', l2, 'l3', l3)

        # feature extractor
        self.fc0 = nn.Linear(in_channels, l1)
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        # readout + head
        self.fc3 = nn.Linear(l3, 128)
        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 l1: int = 8,
                 l2: int = 16,
                 l3: int = 32,
                 n_classes: int = 10):
        super(SimpleCNN, self).__init__()

        print('Setting up CNN with: l1',l1,'l2',l2,'l3',l3)

        # feature exxtractor
        self.conv00 = nn.Conv2d(in_channels=in_channels, out_channels=l1, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv10 = nn.Conv2d(in_channels=l1, out_channels=l2, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv20 = nn.Conv2d(in_channels=l2, out_channels=l3, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2, 2)
        # readout + head
       # self.pool = AdaptiveConcatPool2d(1)
        self.fc0 = nn.Linear(l3 * 400, l3)
     #   self.fc1 = nn.Linear(l3, l3//2)
        self.out = nn.Linear(l3, n_classes)

    def forward(self, x):
        x = F.relu(self.conv00(x))
        #x = self.pool1(x)
        x = F.relu(self.conv10(x))
        #x = self.pool2(x)
        x = F.relu(self.conv20(x))
        #x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
       # x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class SimpleCNNKernel(nn.Module):
    def __init__(self, in_channels: int = 3, l1: int = 5, l2: int = 5, l3: int = 5, n_classes: int = 10):
        super(SimpleCNNKernel, self).__init__()

        print('Setting up CNN with: kernel1', l1, 'kernel2', l2, 'kernel3', l3)

        out_res = 32 - (2*(l1 // 2)) - (2*(l2 // 2)) - (2*(l3 // 2))
        out_res = out_res**2 * 32

        # feature exxtractor
        self.conv00 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=l1)
        self.conv10 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=l2)
        self.conv20 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=l3)
        # readout + head
       # self.pool = AdaptiveConcatPool2d(1)
        self.fc0 = nn.Linear(out_res, l3)
     #   self.fc1 = nn.Linear(l3, l3//2)
        self.out = nn.Linear(l3, n_classes)

    def forward(self, x):
        x = F.relu(self.conv00(x))
        #x = self.pool1(x)
        x = F.relu(self.conv10(x))
        #x = self.pool2(x)
        x = F.relu(self.conv20(x))
        #x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
       # x = F.relu(self.fc1(x))
        x = self.out(x)
        return x
