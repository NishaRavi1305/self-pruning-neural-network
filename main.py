import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------
# Device Setup
# --------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# --------------------------
# Dataset
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False
)

# --------------------------
# Custom Self-Pruning Layer
# --------------------------
class PruningLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores
        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features)
        )

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        effective_weight = self.weight * gates
        return torch.matmul(x, effective_weight.t()) + self.bias

    def sparsity_loss(self):
        gates = torch.sigmoid(self.gate_scores)
        return gates.sum()

    def count_pruned(self, threshold):
        gates = torch.sigmoid(self.gate_scores)
        pruned = (gates < threshold).sum().item()
        total = gates.numel()
        return pruned, total


# --------------------------
# Neural Network
# --------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PruningLinear(3072, 256)
        self.relu = nn.ReLU()
        self.fc2 = PruningLinear(256, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sparsity_loss(self):
        return self.fc1.sparsity_loss() + self.fc2.sparsity_loss()


# --------------------------
# Experiment Function
# --------------------------
def run_experiment(lambda_sparse):

    model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 4
    loss_history = []

    for epoch in range(epochs):

        running_loss = 0.0

        # Dynamic threshold increases each epoch
        threshold = 0.05 + (epoch * 0.02)

        for images, labels in trainloader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            ce_loss = criterion(outputs, labels)
            sparse_loss = lambda_sparse * model.sparsity_loss()

            loss = ce_loss + sparse_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        loss_history.append(avg_loss)

        print(
            f"Lambda {lambda_sparse} | "
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss {avg_loss:.4f} | "
            f"Threshold {threshold:.2f}"
        )

    # --------------------------
    # Testing Accuracy
    # --------------------------
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    # --------------------------
    # Sparsity Calculation
    # --------------------------
    p1, t1 = model.fc1.count_pruned(threshold)
    p2, t2 = model.fc2.count_pruned(threshold)

    pruned = p1 + p2
    total_weights = t1 + t2

    sparsity = 100 * pruned / total_weights

    print(f"Layer1 Sparsity: {100*p1/t1:.2f}%")
    print(f"Layer2 Sparsity: {100*p2/t2:.2f}%")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"Total Pruned: {sparsity:.2f}%")
    print("-" * 50)

    return accuracy, sparsity, loss_history


# --------------------------
# Multi-Lambda Experiments
# --------------------------
lambdas = [0.0001, 0.001, 0.01]

results = []

plt.figure(figsize=(8, 5))

for lam in lambdas:

    accuracy, sparsity, history = run_experiment(lam)

    results.append([lam, accuracy, sparsity])

    plt.plot(history, label=f"λ={lam}")

# --------------------------
# Save Loss Graph
# --------------------------
plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results.png")

# --------------------------
# Save CSV Results
# --------------------------
df = pd.DataFrame(
    results,
    columns=["Lambda", "Accuracy", "Sparsity"]
)

df.to_csv("results.csv", index=False)

# --------------------------
# Save PNG Table
# --------------------------
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center"
)

table.scale(1, 2)
table.auto_set_font_size(False)
table.set_fontsize(10)

plt.title("Experiment Results", pad=20)
plt.savefig("results_table.png", bbox_inches="tight")

print(df)
print("Saved results.png")
print("Saved results.csv")
print("Saved results_table.png")