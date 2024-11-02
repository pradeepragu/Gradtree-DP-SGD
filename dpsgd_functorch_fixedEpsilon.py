import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from functorch import make_functional_with_buffers, vmap, grad
#from torch.func import functional_call, vmap, grad
import os
import numpy as np



import torch
from opacus.accountants import GaussianAccountant
from opacus.accountants import create_accountant
from opacus.accountants.analysis import gdp as privacy_analysis


# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

SEED = 42
set_seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.01
L2_NORM_CLIP = 1.0
NOISE_MULTIPLIER = 1.1
DELTA = 1e-5
#TARGET_EPSILON = 1.0
#SAMPLE_RATE = BATCH_SIZE / len(train_dataset)
# CIFAR10 mean and std for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

# Data preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Data loaders
g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, generator=g)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN().to(device)

# Make the model functional
fmodel, params, buffers =  make_functional_with_buffers(model)

# Loss function
criterion = nn.CrossEntropyLoss()

# Function to compute loss for a single sample
def compute_loss_stateless_model(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    predictions = fmodel(params, buffers, batch)
    loss = criterion(predictions, targets)
    return loss

# Compute gradient for a single sample
ft_compute_grad = grad(compute_loss_stateless_model)

# Compute per-sample gradients for a batch
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
#sample_grads = vmap(grad(your_loss_fn))(params, buffers, images, target)
# Accuracy calculation
def accuracy(preds, labels):
    return (preds == labels).float().mean()


# DP-SGD parameters

TARGET_EPSILON = 8.0  # Privacy budget target (fixed at 1, 3, or 6)

SAMPLE_RATE = BATCH_SIZE / len(train_dataset)




#from opacus.accountants import GaussianAccountant
#from opacus.accountants.analysis import gdp

def get_noise_multiplier(
    target_epsilon, target_delta, sample_rate, epochs, max_noise=10.0, step=0.1
):
    steps = int(epochs / sample_rate)
    accountant = GaussianAccountant()

    noise_multiplier = 0.1  # Start with a small noise multiplier
    while noise_multiplier <= max_noise:
        accountant.history = [(noise_multiplier, sample_rate, steps)]
        try:
            eps = accountant.get_epsilon(delta=target_delta)
            if eps <= target_epsilon:
                return noise_multiplier
        except ValueError:
            pass  # Ignore ValueError and continue increasing noise_multiplier
        noise_multiplier += step

    raise ValueError(f"Could not find a suitable noise multiplier below {max_noise}")

# Calculate the noise multiplier before training
NOISE_MULTIPLIER = get_noise_multiplier(
    target_epsilon=TARGET_EPSILON,
    target_delta=DELTA,
    sample_rate=BATCH_SIZE / len(train_dataset),
    epochs=EPOCHS,
)
print(f"Calculated noise multiplier for fixed privacy budget: {NOISE_MULTIPLIER}")




#print(f"Calculated noise multiplier for fixed privacy budget: {NOISE_MULTIPLIER}")

# Initialize the GaussianAccountant
accountant = GaussianAccountant()

def train(fmodel, params, buffers, train_loader, epoch, device):
    model.train()
    losses = []
    top1_acc = []
    epsilons =[]
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        # Compute per-sample gradients
        sample_grads = ft_compute_sample_grad(params, buffers, images, target)

        # Clip gradients
        total_norm = torch.norm(torch.stack([torch.norm(g.reshape(-1)) for g in sample_grads]))
        clip_coef = torch.clamp(L2_NORM_CLIP / (total_norm + 1e-6), max=1.0)
        clipped_grads = [g * clip_coef for g in sample_grads]

        # Aggregate gradients
        grads = [g.mean(dim=0) for g in clipped_grads]

        # Add noise
        for grad in grads:
            noise = torch.randn_like(grad) * NOISE_MULTIPLIER * L2_NORM_CLIP / BATCH_SIZE
            grad.add_(noise)

        # Manual optimization step
        with torch.no_grad():
            for param, grad in zip(params, grads):
                param.sub_(grad * LEARNING_RATE)

        # Forward pass for loss and accuracy calculation
        outputs = fmodel(params, buffers, images)
        loss = criterion(outputs, target)

        # Calculate accuracy
        preds = torch.argmax(outputs.detach(), dim=1)
        acc = accuracy(preds, target)

        losses.append(loss.item())
        top1_acc.append(acc.item())
        accountant.step(noise_multiplier=NOISE_MULTIPLIER, sample_rate=SAMPLE_RATE)
        
        epsilon = accountant.get_epsilon(delta=DELTA, poisson=True)
        epsilons.append(epsilon)
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], "
                  f'Epsilon: {epsilon:.4f}',
                  f"Loss: {np.mean(losses):.4f}, Acc: {np.mean(top1_acc):.4f}")

    # Update the privacy accountant
    #accountant.step(noise_multiplier=NOISE_MULTIPLIER, sample_rate=SAMPLE_RATE)

    # Compute and return the current epsilon
    eps = accountant.get_epsilon(delta=DELTA)
    return eps



# Testing function
def test(fmodel, params, buffers, test_loader, device):
    model.eval()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            outputs = fmodel(params, buffers, images)
            loss = criterion(outputs, target)
            
            preds = torch.argmax(outputs, dim=1)
            acc = accuracy(preds, target)

            losses.append(loss.item())
            top1_acc.append(acc.item())

    print(f"Test set: Average loss: {np.mean(losses):.4f}, "
          f"Accuracy: {np.mean(top1_acc):.4f}")



def compute_epsilon(steps):
    delta = 1e-5
    eps, best_alpha = accountant.get_privacy_spent(delta=delta)
    return eps


for epoch in tqdm(range(1, EPOCHS + 1), desc="Training"):
    eps = train(fmodel, params, buffers, train_loader, epoch, device)

    #train(fmodel, params, buffers, train_loader, epoch, device)
    test(fmodel, params, buffers, test_loader, device)
    
    # Update the privacy accountant
    #accountant.step(noise_multiplier=NOISE_MULTIPLIER, sample_rate=BATCH_SIZE/len(train_dataset))
    
    # Compute and print the current epsilon
    #eps = compute_epsilon(epoch * len(train_loader))
    print(f"For delta=1e-5, the current epsilon is: {eps:.2f}")

print("Training finished!")
