import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Training settings
lr = 0.0005
momentum = 0.5
log_interval = 10

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device (  # pylint: disable=no-member
    'cuda:1' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """Generator for FashionMNIST dataset."""

    # Extract FashionMNIST data using torchvision datasets
    def read(self, path):
        self.trainset = datasets.FashionMNIST(
            path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor()
            ]))
        self.testset = datasets.FashionMNIST(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ]))
        self.labels = list(self.trainset.classes)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # conv 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # fc1
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        # don't need softmax here since we'll use cross-entropy as activation.
        return t


def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=lr)
    # return optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def get_trainloader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def extract_weights(model):
    weights = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights


def load_weights(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)


def train(model, trainloader, optimizer, epochs):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    epoch_loss = []
    for epoch in range(1, epochs + 1):
        batch_loss = []
        for batch_id, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #############################################################
            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))
            batch_loss.append(float('{:.6f}'.format(loss.item())))
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return sum(epoch_loss) / len(batch_loss)

def test(model, testloader):
    model.to(device)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(  # pylint: disable=no-member
                outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy
