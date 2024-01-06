'''Train CIFAR10 with PyTorch.'''
import torch

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True,)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True,)

print('==>writing train dataset')
train_writer = DatasetWriter(
    './data/cifar_train.beton', 
    {
        'image': RGBImageField(),
        'label': IntField()
    }
)
train_writer.from_indexed_dataset(trainset)

print('==>writing test dataset')
test_writer = DatasetWriter(
    './data/cifar_test.beton', 
    {
        'image': RGBImageField(),
        'label': IntField()
    }
)
test_writer.from_indexed_dataset(testset)