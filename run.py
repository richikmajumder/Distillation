import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define CIFAR-10 data loaders with data augmentation for training
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the pre-trained teacher model
teacher_model = models.resnet18(pretrained=True)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)  # Adjust for CIFAR-10 classes
teacher_model = teacher_model.to('cuda')
teacher_model.eval()  # Set teacher to evaluation mode

# Define the Student Model (simple CNN architecture with dropout)
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 input channels, 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 input channels, 64 output channels
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Fully connected layer with 256 units
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% probability
        self.fc2 = nn.Linear(256, 10)    # Output layer for 10 classes (CIFAR-10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # First conv layer followed by max pooling
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Second conv layer followed by max pooling
        x = x.view(-1, 64 * 8 * 8)  # Flatten before fully connected layer
        x = F.relu(self.dropout(self.fc1(x)))  # Apply dropout here
        x = self.fc2(x)
        return x

# Initialize the student model
student_model = StudentNet().to('cuda')

# Define Knowledge Distillation Loss
def distillation_loss(student_outputs, teacher_outputs, labels, temperature=2, alpha=0.5):
    # Hard Loss: Cross-entropy with true labels
    hard_loss = F.cross_entropy(student_outputs, labels)

    # Soft Loss: KL divergence with teacher outputs
    teacher_soft_outputs = F.softmax(teacher_outputs / temperature, dim=1)
    student_soft_outputs = F.log_softmax(student_outputs / temperature, dim=1)
    soft_loss = F.kl_div(student_soft_outputs, teacher_soft_outputs, reduction='batchmean') * (temperature ** 2)

    # Combined loss
    distill_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    return distill_loss

# Optimizer for the student model
optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)  # Added weight decay for regularization

# Initialize lists to store loss and accuracy for plotting
train_losses = []
train_accuracies = []
test_accuracies = []
distillation_losses = []  # Track distillation loss per epoch

# Evaluation function to compute accuracy
def evaluate_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Function to plot confusion matrix
def plot_confusion_matrix(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Updated training function to track loss and accuracy
def train_student(teacher_model, student_model, train_loader, test_loader, optimizer, num_epochs=50, temperature=2, alpha=0.5, plot_interval=5):
    teacher_model.eval()
    student_model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        distillation_loss_epoch = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')

            # Get teacher and student outputs
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            student_outputs = student_model(images)

            # Calculate distillation loss and track it
            loss = distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha)
            distillation_loss_epoch += loss.item() * images.size(0)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(student_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Record the average losses and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct / total
        epoch_distillation_loss = distillation_loss_epoch / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        distillation_losses.append(epoch_distillation_loss)

        # Evaluate on test set at the end of each epoch
        test_accuracy = evaluate_accuracy(student_model, test_loader)
        test_accuracies.append(test_accuracy)

        # Print epoch metrics
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Distillation Loss: {epoch_distillation_loss:.4f} | Training Accuracy: {epoch_accuracy:.2f}% | Test Accuracy: {test_accuracy:.2f}%")

        # Plot confusion matrix at specific intervals (e.g., every 5 epochs)
        if (epoch + 1) % plot_interval == 0 or epoch == num_epochs - 1:
            plot_confusion_matrix(student_model, test_loader)

# Train the student model and track performance
train_student(teacher_model, student_model, train_loader, test_loader, optimizer, num_epochs=50, temperature=2, alpha=0.5, plot_interval=5)

# Plotting distillation loss and combined accuracy plot for training and test
plt.figure(figsize=(12, 5))

# Plot distillation loss
plt.subplot(1, 2, 1)
plt.plot(range(1, len(distillation_losses) + 1), distillation_losses, label='Distillation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Distillation Loss over Epochs')
plt.legend()

# Combined plot for training and test accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
