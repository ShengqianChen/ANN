'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
'''
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

    
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
'''

classes = ('0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9')

# MNIST 
transform_train = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.1307,), (0.3081,))  
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)


# Model
print('==> Building model..')
net1 = ResNet18()  # ResNet-18
net2 = ResNet34()  # ResNet-34
net1 = net1.to(device)
net2 = net2.to(device)

# Optimizer and Loss function
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=110)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=110)

# For tracking accuracy and loss for both models
train_acc_1, train_loss_1 = [], []
train_acc_2, train_loss_2 = [], []
test_acc_1, test_loss_1 = [], []
test_acc_2, test_loss_2 = [], []

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net1.train()
    net2.train()

    running_loss1 = 0.0
    running_loss2 = 0.0
    correct1 = total1 = 0
    correct2 = total2 = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients for both models
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # Forward pass for ResNet-18
        outputs1 = net1(inputs)
        loss1 = criterion(outputs1, targets)
        loss1.backward()
        optimizer1.step()

        # Forward pass for ResNet-34
        outputs2 = net2(inputs)
        loss2 = criterion(outputs2, targets)
        loss2.backward()
        optimizer2.step()

        # Track accuracy for both models
        _, predicted1 = outputs1.max(1)
        _, predicted2 = outputs2.max(1)
        total1 += targets.size(0)
        correct1 += predicted1.eq(targets).sum().item()
        total2 += targets.size(0)
        correct2 += predicted2.eq(targets).sum().item()

        # Accumulate loss
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()

        progress_bar(batch_idx, len(trainloader), 'Loss1: %.3f | Acc1: %.3f%% (%d/%d) Loss2: %.3f | Acc2: %.3f%% (%d/%d)'
                     % (running_loss1/(batch_idx+1), 100.*correct1/total1, correct1, total1, running_loss2/(batch_idx+1), 100.*correct2/total2, correct2, total2))

    # End of epoch print
    train_acc_1.append(100.*correct1/total1)
    train_loss_1.append(running_loss1 / len(trainloader))
    train_acc_2.append(100.*correct2/total2)
    train_loss_2.append(running_loss2 / len(trainloader))


def test(epoch):
    global best_acc
    net1.eval()
    net2.eval()

    correct1 = total1 = 0
    correct2 = total2 = 0
    running_loss1 = 0.0
    running_loss2 = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass for ResNet-18
            outputs1 = net1(inputs)
            loss1 = criterion(outputs1, targets)

            # Forward pass for ResNet-34
            outputs2 = net2(inputs)
            loss2 = criterion(outputs2, targets)

            # Track accuracy for both models
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            total1 += targets.size(0)
            correct1 += predicted1.eq(targets).sum().item()
            total2 += targets.size(0)
            correct2 += predicted2.eq(targets).sum().item()

            # Accumulate loss
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()

            progress_bar(batch_idx, len(testloader), 'Loss1: %.3f | Acc1: %.3f%% (%d/%d) Loss2: %.3f | Acc2: %.3f%% (%d/%d)'
                     % (running_loss1/(batch_idx+1), 100.*correct1/total1, correct1, total1, running_loss2/(batch_idx+1), 100.*correct2/total2, correct2, total2))

    test_acc_1.append(100.*correct1/total1)
    test_loss_1.append(running_loss1 / len(testloader))
    test_acc_2.append(100.*correct2/total2)
    test_loss_2.append(running_loss2 / len(testloader))

    

    # Save checkpoint if needed
    acc = 100.*correct1/total1
    if acc > best_acc:
        best_acc = acc
        print('Saving..')
        state = {'net1': net1.state_dict(), 'net2': net2.state_dict(), 'acc': acc, 'epoch': epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')

def plot_metrics():
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_acc_1)+1), train_acc_1, label='ResNet-18 Train Acc', color='blue')
    plt.plot(range(1, len(train_acc_2)+1), train_acc_2, label='ResNet-34 Train Acc', color='orange')
    plt.title('Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_loss_1)+1), train_loss_1, label='ResNet-18 Train Loss', color='blue')
    plt.plot(range(1, len(train_loss_2)+1), train_loss_2, label='ResNet-34 Train Loss', color='orange')
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('plots/MNIST_18_34_train_metrics.png')

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(test_acc_1)+1), test_acc_1, label='ResNet-18 Test Acc', color='blue')
    plt.plot(range(1, len(test_acc_2)+1), test_acc_2, label='ResNet-34 Test Acc', color='orange')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_loss_1)+1), test_loss_1, label='ResNet-18 Test Loss', color='blue')
    plt.plot(range(1, len(test_loss_2)+1), test_loss_2, label='ResNet-34 Test Loss', color='orange')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('plots/MNIST_18_34_test_metrics.png')

def save_models(net1, net2, epoch):
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(net1.state_dict(), os.path.join(save_dir, f'model_resnet18_epoch_{epoch}.pth'))
    torch.save(net2.state_dict(), os.path.join(save_dir, f'model_resnet34_epoch_{epoch}.pth'))
    print(f"Models saved at epoch {epoch}")

for epoch in range(20):
    train(epoch)
    test(epoch)
    scheduler1.step()
    scheduler2.step()

#save_models(net1, net2, epoch)
plot_metrics()

