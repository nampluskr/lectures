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
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )        
    def forward(self, x):
        return self.layers(x)

def accuracy(y_pred, y):
    y_pred = y_pred.argmax(dim=1)
    return torch.eq(y_pred, y).float().mean()
```

```python
n_epochs = 10
learning_rate = 1e-3

model = MLP(28*28, 256, 10).to(device)
loss_fn = nn.CrossEntropyLoss()     # with logits
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, n_epochs):
    ## Training
    model.train()
    train_loss, train_acc = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        acc = accuracy(y_pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        train_acc += acc.item()

    ## Validattion        
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            val_loss += loss_fn(y_pred, y).item()
            val_acc += accuracy(y_pred, y).item()
        
    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:3d}/{n_epochs}] "
              f"loss: {train_loss/len(train_loader):.3f} acc: {train_acc/len(train_loader):.3f} | "
              f"val_loss: {val_loss/len(test_loader):.3f} val_acc: {val_acc/len(test_loader):.3f}")
```

```python
from common.trainer import Trainer, EarlyStopping
    
n_epochs = 100
learning_rate = 1e-4

model = MLP(28*28, 256, 10).to(device)
loss_fn = nn.CrossEntropyLoss()     # with logits
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

clf = Trainer(model, optimizer, loss_fn, metrics={"acc": accuracy})
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
stopper = EarlyStopping(patience=3, min_delta=1e-3)

history = clf.fit(train_loader, n_epochs, valid_loader=test_loader,
                  scheduler=scheduler, stopper=stopper)
```

```python
print(clf.evaluate(train_loader))
print(clf.evaluate(test_loader))
```

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3))
ax1.plot(history["loss"], label="loss")
ax1.plot(history["val_loss"], label="val_loss")
ax2.plot(history["acc"], label="acc")
ax2.plot(history["val_acc"], label="val_acc")

ax1.legend()
ax2.legend()
fig.tight_layout()
plt.show()
```
