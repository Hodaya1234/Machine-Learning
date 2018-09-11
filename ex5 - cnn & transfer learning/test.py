from __future__ import print_function, division
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
plt.ion()

batch_size = 64
epochs = 4
learning_rate = 0.004

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
First Model
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.batch1 = nn.BatchNorm1d(120)
        self.batch2 = nn.BatchNorm1d(84)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.batch1(self.fc1(out)))
        out = F.dropout(out, training=self.training)
        out = F.relu(self.batch2(self.fc2(out)))
        out = self.fc3(out)
        return out


def first_model_main():
    lr = [0.001, 0.0007, 0.0005, 0.0003]
    epochs_list = [4, 6, 6, 4]

    classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    net = Net()
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    validation_losses = []
    train_accuracy = []
    validation_accuracy = []

    for i in range(len(epochs_list)):
        optimizer = optim.Adam(net.parameters(), lr=lr[i])
        for epoch in range(epochs_list[i]):
            print('Epoch: {}/{}'.format(epoch+1, epochs_list[i]))

            loss, accuracy = train(net, data_loaders['train'], optimizer, criterion)
            train_losses.append(loss)
            train_accuracy.append(accuracy)

            loss, accuracy = test(net, data_loaders['val'], criterion, validation=True)
            validation_losses.append(loss)
            validation_accuracy.append(accuracy)

    test_loss, test_accuracy, true_classes, estimated_classes = test(net, data_loaders['test'], criterion)

    np.savetxt("test.pred", estimated_classes, fmt='%d', delimiter='\n')
    np.savetxt("real.pred", true_classes, fmt='%d', delimiter='\n')

    plot_result(train_losses, train_accuracy, validation_losses, validation_accuracy, test_loss, test_accuracy)
    plot_confusion_matrix(true_classes, estimated_classes, classes)


"""
Second Model
"""


def create_data(first_model=False):
    if first_model:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    data_size = len(train_set)

    indices = list(range(data_size))
    split = int(data_size * 0.2)

    validation_idx = list(np.random.choice(indices, size=split, replace=False))
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=validation_sampler)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    dataloaders = {'train': train_loader, 'val': validation_loader, 'test': test_loader}
    datasizes = {'train': int(data_size * 0.8), 'val': int(data_size * 0.2), 'test': len(test_set)}
    classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return [dataloaders, datasizes, classes]


def train(net, trainloader, optimizer, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    avg_loss = train_loss / data_sizes['train']
    accuracy = correct / total
    print_info(correct, total, avg_loss, 'Train')
    return [avg_loss, accuracy]


def test(net, test_loader, criterion, validation=False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    true_classes = []
    estimated_classes = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if not validation:
                true_classes.extend(labels.cpu().numpy())
                estimated_classes.extend(predicted.cpu().numpy())
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total
    if validation:
        print_info(correct, total, avg_loss, 'Validation')
        return [avg_loss, accuracy]
    else:
        print_info(correct, total, avg_loss, 'Test')
        return [avg_loss, accuracy, true_classes, estimated_classes]


def plot_result(train_losses, train_accuracy, validation_losses, validation_accuracy, test_loss, test_accuracy):
    fig = plt.figure()

    plt.subplot(121)
    plt.plot(train_losses, label='train')
    plt.plot(validation_losses, label='validation')
    plt.ylabel('Losses')
    plt.xlabel('Epoch')
    plt.title('Losses and Accuracy')
    plt.legend()

    plt.subplot(122)
    plt.plot(train_accuracy, label='train')
    plt.plot(validation_accuracy, label='validation')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    fig.text(0.5, 0.05, 'Test: Accuracy: {} % Loss: {:.4f}'.format(test_accuracy * 100, test_loss), ha='center')

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true_class, estimated_class, classes):
    conf_arr = confusion_matrix(true_class, estimated_class)
    norm_conf = []
    for i in conf_arr:
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    ax.grid(False)
    cb = fig.colorbar(res)
    plt.xticks(range(width), classes[:width])
    plt.yticks(range(height), classes[:height])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def print_info(correct, total, avg_loss, title):
    print('{}: Accuracy: {} ({}/{}) Loss: {:.4f})'
          .format(title, 100 * correct / total, correct, total, avg_loss))


def main():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses = []
    train_accuracy = []
    validation_losses = []
    validation_accuracy = []
    for i in range(epochs):
        print('Epoch: {}/{}'.format(i + 1, epochs))
        loss, accuracy = train(model, data_loaders['train'], optimizer, criterion)
        train_losses.append(loss)
        train_accuracy.append(accuracy)

        loss, accuracy = test(model, data_loaders['val'], criterion, validation=True)
        validation_losses.append(loss)
        validation_accuracy.append(accuracy)

    test_loss, test_accuracy, true_classes, estimated_classes = test(model, data_loaders['test'], criterion)

    plot_result(train_losses, train_accuracy, validation_losses, validation_accuracy, test_loss, test_accuracy)
    plot_confusion_matrix(true_classes, estimated_classes, class_names)


if __name__ == '__main__':
    first = True

    data_loaders, data_sizes, class_names = create_data(first_model=first)
    if first:
        first_model_main()
    else:
        main()
