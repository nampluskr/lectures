### PI Extended DeepONet

- Code: 
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

### Model

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


class Ex_DeepOnet(nn.Module):
    def __init__(self, length, width_trunk, width_branch):
        super().__init__()
        # Branch net
        layers = [length, width_branch, width_branch, width_branch, width_branch, width_trunk * 5]
        self.branch_net = MLP(layer_sizes=layers, activation=nn.ReLU(), final_activation=None)

        # Trunch net
        self.width_trunk = width_trunk
        self.weights = [nn.Linear(1, width_trunk).to(device)]
        for _ in range(self.branch_net.depth - 1):
            self.weights.append(nn.Linear(width_trunk, width_trunk).to(device))
        # self.w0 = nn.Linear(1, width_trunk)
        # self.w1 = nn.Linear(width_trunk, width_trunk)
        # self.w2 = nn.Linear(width_trunk, width_trunk)
        # self.w3 = nn.Linear(width_trunk, width_trunk)
        # self.w4 = nn.Linear(width_trunk, width_trunk)

    def forward(self, x, c):
        x = x.repeat(1, 1, x.shape[1]).permute(0, 2, 1)
        out_branch = self.branch_net(x)

        # for i in range(self.branch_net.depth - 1):
        #     c = self.weights[i] * out_branch[..., i*self.width_trunk:(i+1)*self.width_trunk]
        #     c = F.tanh(c)
        # out_trunk = self.weights[i] * out_branch[..., i*self.width_trunk:(i+1)*self.width_trunk]
        
        for i in range(self.branch_net.depth - 1):
            x = out_branch[:, :, i*self.width_trunk:(i+1)*self.width_trunk]
            c = self.weights[i](c) * x
            c = torch.tanh(c)
        
        x = out_branch[:, :, i*self.width_trunk:(i+1)*self.width_trunk]
        c = self.weights[i](c) * x
        
        # x1 = out_branch[:, :, 0*self.width_trunk:1*self.width_trunk]
        # x2 = out_branch[:, :, 1*self.width_trunk:2*self.width_trunk]
        # x3 = out_branch[:, :, 2*self.width_trunk:3*self.width_trunk]
        # x4 = out_branch[:, :, 3*self.width_trunk:4*self.width_trunk]
        # x5 = out_branch[:, :, 4*self.width_trunk:5*self.width_trunk]

        # c = self.weights[0](c)*x1;  c = F.tanh(c)
        # c = self.weights[1](c)*x2;  c = F.tanh(c)
        # c = self.weights[2](c)*x3;  c = F.tanh(c)
        # c = self.weights[3](c)*x4;  c = F.tanh(c)
        # c = self.weights[4](c)*x5

        y = torch.sum(c, axis=-1)
        return y
```

### Train

```python
width_trunk, width_branch, length = 64, 64, n_points
losses = {}
n_epochs = 1000

model = Ex_DeepOnet(length, width_trunk, width_branch).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

for epoch in range(1, n_epochs + 1):
    model.train()

    losses["ic"] = 0
    losses["eqn"] = 0
    losses["l2"] = 0
    losses["train"] = 0
    for x, c, y in train_loader:
        x, c, y = x.to(device), c.to(device), y.to(device)

        c.requires_grad = True
        y_pred = model(x, c)
        dydx = torch.autograd.grad(y_pred, c, torch.ones_like(y_pred), retain_graph=True, create_graph=True)[0]
        # dydx2 = autograd.grad(dydx, c, torch.ones_like(dydx), retain_graph=True, create_graph=True)[0]

        loss_ic = loss_fn(y_pred[:, 0], torch.zeros_like(y_pred[:, 0]))
        loss_eqn = loss_fn(dydx.squeeze(), x.squeeze()) * 1e-4
        loss_l2 = loss_fn(y_pred.squeeze(), y.squeeze())
        loss = loss_l2 + loss_eqn

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses["ic"] += loss_ic.item()
        losses["eqn"] += loss_eqn.item()
        losses["l2"] += loss_l2.item()
        losses["train"] += loss.item()

    scheduler.step()
    model.eval()
    losses["test"] = 0
    with torch.no_grad():
        for x, c, y in test_loader:
            x, c, y = x.to(device), c.to(device), y.to(device)
            y_pred = model(x, c)
            losses["test"] += loss_fn(y_pred.squeeze(), y.squeeze()).item()

    losses["ic"] /= len(train_loader)
    losses["eqn"] /= len(train_loader)
    losses["l2"] /= len(train_loader)
    losses["train"] /= len(train_loader)
    losses["test"] /= len(test_loader)

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:4d}/{n_epochs}] Train: {losses['train']:.2e} "
              f"(l2: {losses['l2']:.2e} eqn: {losses['eqn']:.2e} ic: {losses['ic']:.2e}) | "
              f"Test: {losses['test']:.2e}")
```

### Evaluation

```python
with torch.no_grad():
    y_pred = model(x_test.to(device), c_test.to(device))

x_test_np = x_test.cpu().detach().numpy().squeeze()
y_test_np = y_test.cpu().detach().numpy().squeeze()
y_pred_np = y_pred.cpu().detach().numpy()
coord = np.linspace(0, 1, n_points)

print(x_test_np.shape, y_test_np.shape, y_pred_np.shape)
idx = 200
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(coord, x_test_np[idx], "k", label="Input")
ax.plot(coord, y_test_np[idx], "k:", label="Exact")
ax.plot(coord, y_pred_np[idx], "r", lw=1, label="Prediction")
ax.legend()
fig.tight_layout()
plt.show()
```
