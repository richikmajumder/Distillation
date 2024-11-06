import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import wide_resnet50_2
from torch.utils.data import DataLoader

# Hyperparameters
learning_rate = 0.01
regularization_param = 0.0001
batch_size = 128
epochs = 200  # Adjust to the desired number of epochs (25 epochs performed for the report analysis)

# CIFAR-10 Data Preparation
transform_cifar = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset_cifar = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_cifar
)
test_dataset_cifar = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_cifar
)

train_loader_cifar = DataLoader(train_dataset_cifar, batch_size=batch_size, shuffle=True)
test_loader_cifar = DataLoader(test_dataset_cifar, batch_size=batch_size, shuffle=False)

# Define the BONN Model using Wide-ResNet for CIFAR-10
class BONN_WideResNet(nn.Module):
    def __init__(self):
        super(BONN_WideResNet, self).__init__()
        self.model = wide_resnet50_2(pretrained=False)
        self.modulation_vector = torch.randn_like(self.model.conv1.weight, requires_grad=True)
        self.mu = torch.mean(self.model.conv1.weight).detach()
        self.sigma = torch.var(self.model.conv1.weight).detach()

    def forward(self, x):
        x = self.binary_conv(x, self.model.conv1)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x

    def binary_conv(self, x, conv_layer):
        binarized_weights = self.modulation_vector * torch.sign(conv_layer.weight)
        return nn.functional.conv2d(x, binarized_weights, bias=conv_layer.bias,
                                    stride=conv_layer.stride, padding=conv_layer.padding)

# Instantiate the model for CIFAR-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cifar = BONN_WideResNet().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_cifar.parameters(), lr=learning_rate,
                      weight_decay=regularization_param, momentum=0.9)

# Lists to store loss and accuracy for plotting
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Training Function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Track accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2f}%')

# Testing Function
def test(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Plot function to visualize loss and accuracy
def plot_metrics():
    epochs_range = range(1, epochs + 1)
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    # Save the plot
    plt.savefig('training_metrics.png')
    plt.show()

# Main loop for training and testing
for epoch in range(epochs):
    train(model_cifar, train_loader_cifar, criterion, optimizer, epoch)
    test(model_cifar, test_loader_cifar, criterion)

# Plot the metrics after training
plot_metrics()
