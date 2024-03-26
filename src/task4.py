import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt

# This is a convenient data reader.
categories = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors.
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images.
])

#### ---- You can change the two lines below to get your own data loaded
train_data = datasets.CIFAR10('./data', download=True, train = True, transform=transform)
validation_data = datasets.CIFAR10('./data', download=True, train = False, transform=transform)

# This samples the 12-th image from the dataset.
# Try changing the index below to see different images in the dataset.
image, category = train_data[24]

# Display the image and its label.
# plt.figure(figsize=(3,3))
# plt.title('This is a %s' % categories[category])
# plt.imshow(image); plt.grid('off');plt.axis('off')
# plt.show()

#### ---- Create your dataloaders here. Read about pytorch DataLoader class.
batch_size = 128

# It additionally has utilities for threaded and multi-parallel data loading.
trainLoader = DataLoader(train_data, batch_size = batch_size,
                         shuffle = True, num_workers = 0)
valLoader = DataLoader(validation_data, batch_size = batch_size,
                       shuffle = False, num_workers = 0)

#### ---- Defining the model and few other hyperparameters

# Defining the model.
from tqdm import tqdm as tqdm
import torch.nn as nn
import torch.optim as optim

learningRate = 1e-2  # Single learning rate for this lab.

# LeNet is French for The Network, and is taken from Yann Lecun's 1998 paper
# on digit classification http://yann.lecun.com/exdb/lenet/
# This was also a network with just two convolutional layers.
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # Convolutional layers.
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Linear layers.
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv1 + ReLU + MaxPooling.
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)

        # Conv2 + ReLU + MaxPooling.
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)


        # This flattens the output of the previous layer into a vector.
        out = out.view(out.size(0), -1)
        # Linear layer + ReLU.
        out = F.relu(self.fc1(out))
        # Another linear layer + ReLU.
        out = F.relu(self.fc2(out))
        # A final linear layer at the end.
        out = self.fc3(out)

        # We will not add nn.LogSoftmax here because nn.CrossEntropy has it.
        # Read the documentation for nn.CrossEntropy.

        return out


# Definition of our network.
classifier = LeNet()

#Definition of our loss.
criterion = nn.CrossEntropyLoss()

# Definition of optimization strategy.
# This optimizer has access to all the parameters in the model.
#
# It can zero all the parameters by doing:
#                                  optimizer.zero_grad()
#
# It can perform an SGD optimization update step in the direction of
# the gradients for each parameters by doing:
#                                  optimizer.step()
#
optimizer = optim.SGD(classifier.parameters(), lr = learningRate)


def train_model(classifier, criterion, optimizer, trainLoader, valLoader, n_epochs = 10):
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(n_epochs):
        # Training phase
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(trainLoader, desc=f'Training Epoch {epoch + 1}/{n_epochs}')

        for images, labels in train_bar:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = classifier(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=(running_loss / total), accuracy=(100 * correct / total))
        
        train_losses.append(running_loss / total)
        train_acc.append(100 * correct / total)

        # epoch_loss = running_loss / len(trainLoader.dataset)
        # print(f"Training Loss: {epoch_loss:.4f}")

        # Validation phase
        classifier.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        val_loss = 0.0

        val_bar = tqdm(valLoader, desc=f'Validation Epoch {epoch + 1}/{n_epochs}')
        with torch.no_grad():  # No gradients required
            for images, labels in val_bar:
                outputs = classifier(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_bar.set_postfix(loss=(val_loss / total), accuracy=(100 * correct / total))
        
        val_losses.append(val_loss / total)
        val_acc.append(100 * correct / total)

        # epoch_accuracy = 100 * correct / total
        # print(f"Validation Accuracy: {epoch_accuracy:.2f}%\n")   
        print(f"Epoch {epoch+1}/{n_epochs} - Training Loss: {train_losses[-1]:.4f}, Training Accuracy: {train_acc[-1]:.2f}%, Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_acc[-1]:.2f}%")
    
    return train_losses, train_acc, val_losses, val_acc



# Call  your training function and
# train_model(classifier, criterion, optimizer,
#             trainLoader, valLoader, n_epochs = 40)


import matplotlib.pyplot as plt

train_losses, train_acc, val_losses, val_acc = train_model(classifier, criterion, optimizer, trainLoader, valLoader, n_epochs=100)

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

