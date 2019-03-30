from types import FunctionType
from fastai.vision import create_cnn, ImageDataBunch
from fastai.train import AdamW

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from delve import CheckLayerSat
from models import SimpleFCNet, SimpleCNN, SimpleCNNKernel


from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)

def train_set_cifar(transform, batch_size):
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return train_loader

def test_set_cifar(transform, batch_size):
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return test_loader

def train(network, dataset, test_set, logging_dir):

    network.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(network.parameters())
    stats = CheckLayerSat(logging_dir, network)
    test_stats = CheckLayerSat(logging_dir.replace('train','valid'), network)

    train_loss = 0
    epoch_acc = 0
    thresh = 0.95
    epoch = 0
    total = 0
    correct = 0
    while epoch <= 20:
        print('Start Training Epoch', epoch, '\n')

        epoch_acc = 0
        total = 0
        correct = 0
        for i, data in enumerate(dataset):
            step = epoch*len(dataset) + i
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(i,'of', len(dataset),'acc:', correct/total)
            # display layer saturation levels
        stats.saturation()
        test(network, test_set, criterion, test_stats, epoch)
        epoch_acc = correct / total
        print('Epoch', epoch, 'finished', 'Acc:', epoch_acc, 'Loss:', train_loss / step,'\n')
        stats.add_scalar('loss', train_loss / step, epoch)  # optional
        stats.add_scalar('acc', epoch_acc, epoch)  # optional
        epoch += 1
    stats.close()
    test_stats.close()

    return criterion


def test(network, dataset, criterion, stats, epoch):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataset):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 2000 == 1999:  # print every 2000 mini-batches
                print(batch_idx,'of', len(dataset),'acc:', correct/total)
        stats.saturation()
        print('Test finished', 'Acc:', correct / total, 'Loss:', test_loss / len(dataset),'\n')
        stats.add_scalar('test_loss', test_loss / len(dataset), epoch)  # optional
        stats.add_scalar('test_acc', correct/total, epoch)  # optional
    return

def execute_experiment(network: nn.Module, in_channels: int, n_classes: int, l1: int, l2: int , l3: int, train_set: FunctionType, test_set: FunctionType):

    print('Experiment has started')

    batch_size = 64

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    for l1_config in l1:
        for l2_config in l2:
            for l3_config in l3:
                print('Creating Network')

                net = network(in_channels=in_channels,
                        l1=l1_config,
                        l2=l2_config,
                        l3=l3_config,
                        n_classes=n_classes)
                print('Network created')

                train_loader = train_set(transform, batch_size)
                test_loader = test_set(transform, batch_size)


                print('Datasets fetched')
                train(net, train_loader, test_loader, 'train_run_{}_{}_{}'.format(l1_config, l2_config, l3_config))

if '__main__' == __name__:

    configCNN_cifar = {
        'network': SimpleCNN,
        'in_channels': 3,
        'n_classes': 10,
        'l1' : [4, 16, 64],
        'l2' : [8, 32, 128],
        'l3' : [16, 64, 256],
        'train_set': train_set_cifar,
        'test_set': test_set_cifar
    }

    configCNNKernel_cifar = {
        'network': SimpleCNNKernel,
        'in_channels': 3,
        'n_classes': 10,
        'l1': [3, 5, 7],
        'l2': [3, 5, 7],
        'l3': [3, 5, 7],
        'train_set': train_set_cifar,
        'test_set': test_set_cifar
    }

    configFCN_cifar = {
        'network': SimpleFCNet,
        'in_channels': 32*32*3,
        'n_classes': 10,
        'l1' : [4*3*3, 16*3*3, 64*3*3],
        'l2' : [8*3*3, 32*3*3, 128*3*3],
        'l3' : [16*3*3, 64*3*3, 256*3*3],
        'train_set': train_set_cifar,
        'test_set': test_set_cifar
    }

    execute_experiment(**configCNNKernel_cifar)