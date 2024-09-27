#privacy Budget 1
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

# Data Preparation
# Load dataset
data_path = '/content/car.data'
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv(data_path, names=columns)

# Convert categorical data to numerical
df = pd.get_dummies(df, columns=["buying", "maint", "doors", "persons", "lug_boot", "safety"])

# Split features and labels
X = df.drop('class', axis=1)
y = pd.get_dummies(df['class'])

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Define a Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training with DP-SGD and Opacus
def train_model_with_dp_sgd():
    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    epochs = 30
    target_epsilon = 1.0  # Privacy budget
    delta = 1e-5          # Delta value
    
    # Prepare DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = SimpleNN(input_size=X_train.shape[1], num_classes=y_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Compute the noise multiplier explicitly
    sample_rate = batch_size / len(X_train)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=epochs,
        accountant="prv"
    )

    print(f"Calculated noise multiplier: {noise_multiplier}")

    # Set up Opacus PrivacyEngine using make_private
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=1.0,
    )

    # Tracking loss, accuracy, and privacy metrics
    losses = []
    accuracies = []
    epsilons = []

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(target, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss /= len(train_loader)
        accuracy = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(accuracy)

        # Track privacy loss after each epoch
        epsilon = privacy_engine.get_epsilon(delta)
        epsilons.append(epsilon)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, (ε = {epsilon:.2f}, δ = {delta})")
        
        # Stop training when target epsilon is reached
        if epsilon > target_epsilon:
            print("Reached target privacy budget. Stopping training.")
            break

    return model, losses, accuracies, epsilons

# Train the model and capture loss, accuracy, and epsilon
model, losses, accuracies, epsilons = train_model_with_dp_sgd()

# Plotting the Training Loss, Accuracy, and Privacy Budget (Epsilon)
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
plt.plot(accuracies, label='Training Accuracy', color='green')
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

# Model Evaluation on Test Set
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    loss = nn.CrossEntropyLoss()(outputs, y_test)
    
    # Calculate test accuracy
    _, predicted = torch.max(outputs.data, 1)
    _, labels = torch.max(y_test, 1)
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / labels.size(0)

    print(f"Test Loss: {loss.item():.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
