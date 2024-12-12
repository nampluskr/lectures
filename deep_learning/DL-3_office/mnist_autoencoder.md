```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")
```

```python
import numpy as np
import os
import gzip

def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

data_dir = r"..\data\mnist"
# data_dir = r"..\data\fashion_mnist"

x_train = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
y_train = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
x_test = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
y_test = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")

print(f">> Train images: {x_train.shape}, {x_train.dtype}")
print(f">> Train labels: {y_train.shape}, {y_train.dtype}")
print(f">> Test images:  {x_test.shape}, {x_test.dtype}")
print(f">> Test labels:  {y_test.shape}, {y_test.dtype}")
```

```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return image, label

x_train = x_train.reshape(-1, 28*28) / 255
x_test = x_test.reshape(-1, 28*28) / 255
    
train_loader = DataLoader(Dataset(x_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(Dataset(x_test, y_test), batch_size=64, shuffle=True)

x, y = next(iter(train_loader))
print(f">> x: {x.shape}, {x.dtype}, min={x.min()}, max={x.max()}")
print(f">> y: {y.shape}, {y.dtype}, min={y.min()}, max={y.max()}")
```

```python
class MLPEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),)

    def forward(self, x):
        return self.encoder(x)

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 28*28),)

    def forward(self, z):
        return self.decoder(z)
    
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x
    
def binary_accuracy(x_pred, x_true):
    return torch.eq(x_pred.round(), x_true.round()).float().mean()
```

```python
from common.trainer import TrainerAE, EarlyStopping

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 2
n_epochs = 50
learning_rate = 1e-4

encoder = MLPEncoder(latent_dim)
decoder = MLPDecoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
loss_fn = nn.BCELoss()
metrics = {"acc": binary_accuracy}
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

ae = TrainerAE(model, optimizer, loss_fn, metrics=metrics)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
stopper = EarlyStopping(patience=3, min_delta=1e-3)

history = ae.fit(train_loader, n_epochs, valid_loader=test_loader,
                 scheduler=scheduler, stopper=stopper, step_size=1)
```

```python
import matplotlib.pyplot as plt

x, y = next(iter(test_loader))

model.eval()
with torch.no_grad():
    x, y = x.to(device), y.to(device)
    x_pred = model(x)

x = x.cpu().cpu().detach().numpy().reshape(-1, 28, 28)
x_pred = x_pred.cpu().detach().numpy().reshape(-1, 28, 28)

n_images = 12
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 2))
ax1.imshow(np.hstack(x[:n_images]), cmap="gray_r")
ax2.imshow(np.hstack(x_pred[:n_images]), cmap="gray_r")
for ax in (ax1, ax2):
    ax.set_axis_off()
fig.tight_layout()
plt.show()
```

```python
x = torch.tensor(x_train).float()

model.eval()
with torch.no_grad():
    x = x.to(device)
    z = model.encoder(x)

z = z.cpu().detach().numpy()
y = y_train
z1, z2 = z.T

fig, ax = plt.subplots(figsize=(4, 4))
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    ax.scatter(z1[y == i], z2[y == i], alpha=0.5, s=2, label=i)

ax.legend()
fig.tight_layout()
plt.show()
```

```python
class CNNEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, (3, 3), stride=2, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(64*7*7, 256),
            nn.Tanh(),
            nn.Linear(256, latent_dim),)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        return self.encoder(x)


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 64*7*7),
            nn.Tanh(),
            nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),
            nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 1, (3, 3), stride=1, padding=1),)

    def forward(self, x):
        x = self.decoder(x)
        return x.view(-1, 28*28)
```

```python
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 2
n_epochs = 50
learning_rate = 1e-3

encoder = CNNEncoder(latent_dim)
decoder = CNNDecoder(latent_dim)
model = AutoEncoder(encoder, decoder).to(device)
loss_fn = nn.BCELoss()
metrics = {"acc": binary_accuracy}
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

ae = TrainerAE(model, optimizer, loss_fn, metrics=metrics)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
stopper = EarlyStopping(patience=3, min_delta=1e-3)

history = ae.fit(train_loader, n_epochs, valid_loader=test_loader,
                 scheduler=scheduler, stopper=stopper, step_size=1)
```

```python
import matplotlib.pyplot as plt

x, y = next(iter(test_loader))

model.eval()
with torch.no_grad():
    x, y = x.to(device), y.to(device)
    x_pred = model(x)

x = x.cpu().cpu().detach().numpy().reshape(-1, 28, 28)
x_pred = x_pred.cpu().detach().numpy().reshape(-1, 28, 28)

n_images = 12
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 2))
ax1.imshow(np.hstack(x[:n_images]), cmap="gray_r")
ax2.imshow(np.hstack(x_pred[:n_images]), cmap="gray_r")
for ax in (ax1, ax2):
    ax.set_axis_off()
fig.tight_layout()
plt.show()
```

```python
x = torch.tensor(x_train).float()

model.eval()
with torch.no_grad():
    x = x.to(device)
    z = model.encoder(x)

z = z.cpu().detach().numpy()
y = y_train
z1, z2 = z.T

fig, ax = plt.subplots(figsize=(4, 4))
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    ax.scatter(z1[y == i], z2[y == i], alpha=0.5, s=2, label=i)

ax.legend()
fig.tight_layout()
plt.show()
```
