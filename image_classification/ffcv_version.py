'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from typing import List
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, RandomResizedCrop
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

import os
import argparse

from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

# Data
print('==> Preparing data..')
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def make_trainloaders(train_dataset_path):
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda:0')), Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder(),  ToTensor(),
            ToDevice(torch.device('cuda:0'), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float32),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),]
    
    return Loader(train_dataset_path, batch_size=128, num_workers=2,
                               order=OrderOption.RANDOM, drop_last=(True),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

def make_testloaders(test_dataset_path):
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda:0')), Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder(),  ToTensor(),
            ToDevice(torch.device('cuda:0'), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float32),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),]
    
    return Loader(test_dataset_path, batch_size=100, num_workers=2,
                               order=OrderOption.SEQUENTIAL, drop_last=(False),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

trainset_path = 'data/cifar_train.beton'
testset_path = 'data/cifar_test.beton'
trainloader = make_trainloaders(trainset_path)
testloader = make_testloaders(testset_path)

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
@timeit
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


@timeit
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return acc

for epoch in range(start_epoch, start_epoch+10):

    train(epoch)
    test(epoch)
    scheduler.step()
