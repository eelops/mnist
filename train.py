import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256, bias=False)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.bn(self.fc1(x)))
        return self.fc2(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP().to(device)

tf = transforms.ToTensor()
train_set = datasets.MNIST('./data', train=True , download=True, transform=tf)
test_set  = datasets.MNIST('./data', train=False, download=True, transform=tf)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set , batch_size=128)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            pred = model(imgs).argmax(dim=1)
            correct += (pred == labels).sum().item()
    print(f'epoch {epoch} acc {correct / len(test_set):.4f}')

        
