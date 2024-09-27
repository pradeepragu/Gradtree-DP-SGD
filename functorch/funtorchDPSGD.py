#manually adjust noise multipliier to get realistic privacy budget, check readme file for the noise multipler values

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import math

# Set random seed for reproducibility
torch.manual_seed(0)

# Load and preprocess the dataset
car_data = pd.read_csv('/content/car.data', header=None)

# Preprocessing steps:
# Assume the last column is the target, and the rest are features.
features = car_data.iloc[:, :-1]
targets = car_data.iloc[:, -1]

# Convert categorical features and targets into numerical values
encoder = LabelEncoder()
features = features.apply(encoder.fit_transform)
targets = encoder.fit_transform(targets)

# Scale the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create a custom dataset class
class CarDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoader for training and testing
batch_size = 64
train_dataset = CarDataset(X_train_tensor, y_train_tensor)
test_dataset = CarDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the NN model for tabular data
class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Loss function
def loss_fn(predictions, targets):
    return F.cross_entropy(predictions, targets)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, optimizer, and other hyperparameters
input_dim = X_train.shape[1]
output_dim = len(set(y_train))
model = SimpleMLP(input_dim, output_dim).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# DP-SGD parameters
max_grad_norm = 1.0
target_epsilon = 1.0  # Privacy budget target
delta = 1e-5          # Delta value
noise_multiplier = 20  # noise multiplier

# Helper functions for DP-SGD
def compute_per_sample_gradients(model, loss_fn, data, targets):
    model.zero_grad()
    outputs = model(data)
    loss = loss_fn(outputs, targets)
    per_sample_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    return per_sample_grads

def clip_gradients(gradients, max_norm):
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), p=2) for g in gradients]))
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    return [g.detach() * clip_coef_clamped for g in gradients]

def add_noise(gradients, noise_multiplier, max_norm):
    noised_gradients = []
    for grad in gradients:
        noise = torch.normal(0, noise_multiplier * max_norm / batch_size, grad.shape, device=grad.device)
        noised_gradients.append(grad + noise)
    return noised_gradients

def train_step(model, optimizer, data, targets):
    per_sample_grads = compute_per_sample_gradients(model, loss_fn, data, targets)
    clipped_grads = clip_gradients(per_sample_grads, max_grad_norm)
    noised_grads = add_noise(clipped_grads, noise_multiplier, max_grad_norm)

    optimizer.zero_grad()
    for param, noised_grad in zip(model.parameters(), noised_grads):
        param.grad = noised_grad
    optimizer.step()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy

def compute_epsilon(steps, noise_multiplier, batch_size, dataset_size, delta):
    """Compute epsilon using a simplified formula."""
    q = batch_size / dataset_size
    T = steps * q
    c = math.sqrt(2 * math.log(1.25 / delta))
    return c * math.sqrt(T) / noise_multiplier

# Training loop
steps = 0
losses = []
epsilons = []
train_accuracies = []

for epoch in range(50):  # Specify number of epochs here
    model.train()
    epoch_loss = 0
    correct_train = 0
    total_train = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Perform a training step
        train_step(model, optimizer, data, target)
        steps += 1

        # Compute loss for the current batch
        outputs = model(data)
        loss = loss_fn(outputs, target)
        epoch_loss += loss.item()

        # Compute the number of correct predictions in the training batch
        pred = outputs.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
        correct_train += pred.eq(target.view_as(pred)).sum().item()
        total_train += target.size(0)

    # Compute average loss over the training data
    losses.append(epoch_loss / len(train_loader))

    # Calculate training accuracy
    train_accuracy = 100. * correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Evaluate on test set
    test_loss, test_accuracy = test(model, device, test_loader)

    print(f'Epoch {epoch+1}: Train Loss: {epoch_loss / len(train_loader):.4f}, '
          f'Train Accuracy: {train_accuracy:.2f}%, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Compute privacy spent
    epsilon = compute_epsilon(steps, noise_multiplier, batch_size, len(train_dataset), delta)
    epsilons.append(epsilon)
    print(f"Current privacy guarantee: ε = {epsilon:.2f} at δ = {delta}")

    # Stop if the current epsilon exceeds the target epsilon
    if epsilon >= target_epsilon:
        print("Target privacy budget reached. Stopping training.")
        break

# Final privacy guarantee
epsilon = compute_epsilon(steps, noise_multiplier, batch_size, len(train_dataset), delta)
print(f"Final privacy guarantee: ε = {epsilon:.2f} at δ = {delta}")

# Plotting the training loss, training accuracy, and epsilon
plt.figure(figsize=(18, 5))

# Plot the training loss
plt.subplot(1, 3, 1)
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# Plot the training accuracy
plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Over Epochs')
plt.legend()

# Plot the privacy budget (epsilon)
plt.subplot(1, 3, 3)
plt.plot(epsilons, label='Epsilon', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Epsilon (Privacy Budget)')
plt.title('Epsilon Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
