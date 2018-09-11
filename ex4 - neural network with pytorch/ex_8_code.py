import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
batch_size = 64
epochs = 10
lr = 0.005
drop_out = 0.4
image_size = 28*28
first_layer = 100
second_layer = 50
output_size = 10


class ConvolutionNet(nn.Module):
    def __init__(self, size=image_size):
        super(ConvolutionNet, self).__init__()
        self.image_size = size
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, second_layer)
        self.fc2 = nn.Linear(second_layer, output_size)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DropoutNet(nn.Module):
    def __init__(self, size=image_size):
        super(DropoutNet, self).__init__()
        self.image_size = size
        self.fc0 = nn.Linear(size, first_layer)
        self.fc1 = nn.Linear(first_layer, second_layer)
        self.fc2 = nn.Linear(second_layer, output_size)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(self.dropout(x)))
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class NormNet(nn.Module):
    def __init__(self, size=image_size):
        super(NormNet, self).__init__()
        self.image_size = size
        self.fc0 = nn.Linear(size, first_layer)
        self.fc1 = nn.Linear(first_layer, second_layer)
        self.fc2 = nn.Linear(second_layer, output_size)
        self.batch1 = nn.BatchNorm1d(first_layer)
        self.batch2 = nn.BatchNorm1d(second_layer)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.batch1(self.fc0(x)))
        x = F.relu(self.batch2(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LinearNet(nn.Module):
    def __init__(self, size=image_size):
        super(LinearNet, self).__init__()
        self.image_size = size
        self.fc0 = nn.Linear(size, first_layer)
        self.fc1 = nn.Linear(first_layer, second_layer)
        self.fc2 = nn.Linear(second_layer, output_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, train_loader, optimizer):
    model.train()
    correct = 0
    average_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        average_loss += F.nll_loss(output, target)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    num_batches = len(train_loader)
    size = num_batches * batch_size
    average_loss /= num_batches
    accuracy = 100. * correct / size
    print('Train:\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        average_loss, correct, size, accuracy))
    return [average_loss.item(), accuracy]


def validate(model, test_loader):
    model.eval()
    average_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            average_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    size = len(test_loader.sampler)
    average_loss /= size
    accuracy = 100. * correct / size
    print('Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        average_loss, correct, size, accuracy))
    return [average_loss, accuracy]


def test(model, test_loader):
    real_targets = []
    targets = []
    model.eval()
    average_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            average_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            targets.append(pred)
            real_targets.append(target)
    size = len(test_loader.sampler)
    average_loss /= size
    accuracy = 100. * correct / size
    print('Validation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        average_loss, correct, size, accuracy))

    np.savetxt("test.pred", targets, fmt='%d', delimiter='\n')
    np.savetxt("real.pred", real_targets, fmt='%d', delimiter='\n')
    return [average_loss, accuracy]


def create_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    data_size = len(train_set)

    indices = list(range(data_size))
    split = int(data_size * 0.2)

    validation_idx = list(np.random.choice(indices, size=split, replace=False))
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(train_set, batch_size=1, sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data', train=False, transform=transform),
                                              batch_size=1, shuffle=False)

    return [train_loader, validation_loader, test_loader]


def plot_losses(train_losses, validation_losses, title):
    plt.plot(train_losses, 'r')
    plt.plot(validation_losses, 'b')
    plt.title(title)
    plt.ylabel('Losses: Train - red, Validation - blue')
    plt.xlabel('epoch')
    plt.show()


def main():
    train_loader, validation_loader, test_loader = create_data()

    model = ConvolutionNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracy = []
    validation_losses = []
    validation_accuracy = []

    for epoch in range(1, epochs + 1):
        loss, accuracy = train(model, train_loader, optimizer)
        train_losses.append(loss)
        train_accuracy.append(accuracy)

        loss, accuracy = validate(model, validation_loader)
        validation_losses.append(loss)
        validation_accuracy.append(accuracy)

    test_loss, test_accuracy = test(model, test_loader)

    print("Train:")
    print(np.mean(train_losses))
    print(np.mean(train_accuracy))

    print("Validation:")
    print(np.mean(validation_losses))
    print(np.mean(validation_accuracy))

    print("Test:")
    print(np.mean(test_loss))
    print(np.mean(test_accuracy))

    plot_losses(train_losses, validation_losses, 'Convolution')


if __name__ == '__main__':
    main()
