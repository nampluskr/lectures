### DeepONet Varations

- Code: 
  - https://github.com/hl4220/Extended-Physics-Informed-Neural-Operator/blob/main/Anti-derivative/PI_deepOnet_PoD.ipynb
  - https://github.com/hl4220/Extended-Physics-Informed-Neural-Operator/blob/main/Anti-derivative/PI_ExdeepOnet_Anti_dev.ipynb

- Data: ETH Zurich's course on "Deep Learning in Scienfitic Computing"
  - https://github.com/mroberto166/CAMLab-DLSCTutorials/raw/main/antiderivative_aligned_train.npz
  - https://github.com/mroberto166/CAMLab-DLSCTutorials/raw/main/antiderivative_aligned_test.npz

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # current gpu seed
torch.cuda.manual_seed_all(seed) # All gpu seed
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = False  # True로 하면 gpu에 적합한 알고리즘을 선택함.
```

### Data

```python
## Data
data_train = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
branch_inputs_train = data_train["X"][0]    # (n_samples, n_points)     Fs
trunk_inputs_train = data_train["X"][1]     # (n_points, 1)             coor
outputs_train = data_train["y"]             # (n_samples, n_points)     Y

data_test = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
branch_inputs_test = data_test["X"][0]      # (n_samples, n_points)
trunk_inputs_test = data_test["X"][1]       # (n_points, 1)
outputs_test = data_test["y"]               # (n_samples, n_points)

print(">> Train Data:", branch_inputs_train.shape, trunk_inputs_train.shape, outputs_train.shape)
print(">> Test Data: ", branch_inputs_test.shape, trunk_inputs_test.shape, outputs_test.shape)
```

```python
## Dataset and Dataloader
from torch.utils.data import TensorDataset, DataLoader

n_train = branch_inputs_train.shape[0]
n_test = branch_inputs_test.shape[0]
n_points = branch_inputs_train.shape[1]

x_train = torch.tensor(branch_inputs_train).float().unsqueeze(-1)
c_train = torch.linspace(0, 1, n_points)[None, :, None].repeat(n_train, 1, 1)
y_train = torch.tensor(outputs_train).float().unsqueeze(-1)

x_test = torch.tensor(branch_inputs_test).float().unsqueeze(-1)
c_test = torch.linspace(0, 1, n_points)[None, :, None].repeat(n_test, 1, 1)
y_test = torch.tensor(outputs_test).float().unsqueeze(-1)

print(">> Train Data:", x_train.shape, c_train.shape, y_train.shape)
print(">> Test Data: ", x_test.shape, c_test.shape, y_test.shape)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

batch_size = 16
train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=batch_size, shuffle=False)

x, c, y = next(iter(train_loader))
print(x.shape, c.shape, y.shape)
```

### Models

```python
class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU(), final_activation=None):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.depth = len(layer_sizes) - 1

        layers = []
        for i in range(self.depth - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        if final_activation is not None:
            layers.append(final_activation)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
```

```python
class DeepOnetVanila(nn.Module):
    def __init__(self, width_trunk, width_branch, length):
        super().__init__()

        # Brunch net
        branch_layers = [length, width_branch, width_branch, width_branch, width_branch, width_trunk]
        self.branch_net = MLP(branch_layers, activation=nn.ReLU())

        # Trunk net
        truck_layers = [1, width_trunk, width_trunk, width_trunk, width_trunk, width_branch]
        self.trunk_net = MLP(truck_layers, activation=nn.Tanh(), final_activation=nn.Tanh())

    def forward(self, x, c):
        x = x.repeat(1, 1, x.shape[1]).permute(0, 2, 1)
        out_branch = self.branch_net(x)
        out_trunk = self.trunk_net(c)

        out = torch.sum(out_branch * out_trunk, axis=-1)
        return out

class DeepOnetPOU(nn.Module):
    def __init__(self, width_trunk, width_branch, length):
        super().__init__()

        # Brunch net
        branch_layers = [length, width_branch, width_branch, width_branch, width_branch, width_trunk]
        self.branch_net = MLP(layer_sizes=branch_layers)

        # Trunk net
        truck_layers = [1, width_trunk, width_trunk, width_trunk, width_trunk, width_branch]
        self.trunk_net = MLP(layer_sizes=truck_layers)

    def forward(self, x, c):
        x = x.repeat(1, 1, x.shape[1]).permute(0, 2, 1)
        out_branch = self.branch_net(x)
        out_trunk = self.trunk_net(c)

        out = torch.sum(out_branch * out_trunk, axis=-1)
        pou = torch.sum(out_trunk, axis=-1)     # partition of unitiy
        return out, pou
```

### Train - Vanlia DeepONet

```python
model = DeepOnetVanila(width_trunk=64, width_branch=64, length=100).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
n_epochs = 1000

for epoch in range(1, n_epochs + 1):
    model.train()
    loss_mse = 0
    train_loss = 0
    for x, c, y in train_loader:
        x, c, y = x.to(device), c.to(device), y.to(device)

        y_pred = model(x, c)
        mse = loss_fn(y_pred.squeeze(), y.squeeze())
        loss =  mse

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_mse += mse.item()
        train_loss += loss.item()

    scheduler.step()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, c, y in test_loader:
            x, c, y = x.to(device), c.to(device), y.to(device)

            y_pred = model(x,c)
            test_loss += loss_fn(y_pred.squeeze(), y.squeeze()).item()

    loss_mse /= len(train_loader)
    test_loss /= len(test_loader)

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:3d}/{n_epochs}] train_loss: {train_loss:.2e} (mse: {loss_mse:.2e}) | test_loss: {test_loss:.2e}")
```

### Train: DeepONet with POU

```python
model = DeepOnetPOU(width_trunk=64, width_branch=64, length=100).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
n_epochs = 1000

for epoch in range(1, n_epochs + 1):
    model.train()
    loss_mse = 0
    loss_pou = 0
    train_loss = 0
    for x, c, y in train_loader:
        x, c, y = x.to(device), c.to(device), y.to(device)

        y_pred, pou = model(x, c)
        mse = loss_fn(y_pred.squeeze(), y.squeeze())
        pou = loss_fn(pou, torch.ones_like(pou))

        loss =  mse + pou
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_mse += mse.item()
        loss_pou += pou.item()
        train_loss += loss.item()

    scheduler.step()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, c, y in test_loader:
            x, c, y = x.to(device), c.to(device), y.to(device)

            y_pred, y_pou = model(x,c)
            test_loss += loss_fn(y_pred.squeeze(), y.squeeze()).item()

    loss_mse /= len(train_loader)
    loss_pou /= len(train_loader)
    test_loss /= len(test_loader)

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:3d}/{n_epochs}] train_loss: {train_loss:.2e} (mse: {loss_mse:.2e} pou: {loss_pou:.2e}) | test_loss: {test_loss:.2e}")
```
