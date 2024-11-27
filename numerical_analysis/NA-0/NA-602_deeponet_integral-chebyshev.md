## Deep O Net Tutorial

- Demo for Pytorch Implementaion of Data Driven and Physics Informed Deep O nets
  - https://github.com/JohnCSu/DeepONet_Pytorch_Demo

- Code: Operator Learning and Implementation in Pytorch
  - https://johncsu.github.io/DeepONet_Demo/

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```

```python
from numpy.polynomial.chebyshev import chebval
from scipy.integrate import cumulative_trapezoid

def get_data(x, n_samples, degree=20, M=5, seed=42):
    np.random.seed(seed)
    n_points = x.shape[0]
    u = np.empty((n_samples, n_points))
    Gu = np.empty((n_samples, n_points))

    for i in range(n_samples):
        coeff = (np.random.rand(degree + 1) - 0.5)*2*np.abs(M)
        u[i] = chebval(np.linspace(-1, 1, n_points), coeff)
        Gu[i] = cumulative_trapezoid(u[i], x, initial=0)
        
    return u, Gu

n_points, n_samples = 201, 1000
y = np.linspace(0, 2, n_points)
u, Gu = get_data(y, n_samples, seed=42)
print(f"y: {y.shape} u: {u.shape} Gu: {Gu.shape}")

idx = 10
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(y, u[idx], 'k', label='u')
ax.plot(y, Gu[idx], 'r', label='Gu')
ax.legend(loc="upper right")
ax.grid(color='k', ls=':', lw=0.5)
fig.tight_layout()
plt.show()
```

```python
# Model
class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.Tanh(), final_activation=None):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        if final_activation is not None:
            layers.append(final_activation)

        self.layers = nn.Sequential(*layers)

    def init(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # nn.init.xavier_normal_(layer.weight.data)
                nn.init.kaiming_uniform_(layer.weight.data)
        return self

    def forward(self, x):
        return self.layers(x)

class DeepONet1D(nn.Module):
    def __init__(self, branch_net, trunk_net):
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.bias = nn.Parameter(torch.zeros((1,)))
        
    def forward(self, x_branch, x_trunk):
        out_branch = self.branch_net(x_branch)  # (n_batch, interact_dim)
        out_trunk = self.trunk_net(x_trunk)     # (n_points, interact_dim)
        return torch.mm(out_branch, out_trunk.t()) + self.bias
```

```python
hidden_size, interact_size = 100, 50
n_epochs = 50000
learning_rate = 1e-3

branch_layers = [n_points, hidden_size, hidden_size, interact_size]
branch_net = MLP(branch_layers, activation=nn.ReLU())
trunk_layers = [1, hidden_size, hidden_size, interact_size]
trunk_net = MLP(trunk_layers, activation=nn.ReLU(), final_activation=nn.ReLU())
model = DeepONet1D(branch_net, trunk_net).to(device)

## Train
x_branch = torch.tensor(u).float().to(device) # (n_samples, n_points)
x_trunk = torch.tensor(y).float().view(-1, 1).to(device)   # (n_points, 1)
out_train = torch.tensor(Gu).float().to(device)      # (n_samples, n_points)
print(x_branch.shape, x_trunk.shape, out_train.shape)

loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(1, n_epochs + 1):
    model.train()
    out = model(x_branch, x_trunk)
    loss = loss_fn(out, out_train)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % (n_epochs // 5) == 0:
        print(f"[{epoch}/{n_epochs}] loss: {loss.item():.2e}")
        
optimizer = optim.Adam(model.parameters(), lr=learning_rate*0.1)
for epoch in range(1, n_epochs + 1):
    model.train()
    out = model(x_branch, x_trunk)
    loss = loss_fn(out, out_train)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % (n_epochs // 5) == 0:
        print(f"[{epoch}/{n_epochs}] loss: {loss.item():.2e}")
```

```python
## Tetst
a = 15
y_test_np = np.linspace(0, 2, n_points)
u_test_np = np.sin(a * y_test_np)
Gu_test_np = -(np.cos(a * y_test_np) - 1) / a

x_branch_test = torch.tensor(u_test_np).float().view(1, -1).to(device)
x_trunk_test = torch.tensor(y_test_np).float().view(-1, 1).to(device)
out_test = torch.tensor(Gu_test_np).float().view(1, -1).to(device)

print(x_branch_test.shape, x_trunk_test.shape, out_test.shape)

with torch.no_grad():
    model.eval()
    out = model(x_branch_test, x_trunk_test)
    
out_np = out.cpu().detach().numpy().flatten()

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(y_test_np, Gu_test_np, 'k:', label='Exact')
ax.plot(y_test_np, out_np, 'r', lw=1, label='Prediction')
ax.legend(loc="upper right")
ax.grid(color='k', ls=':', lw=0.5)
fig.tight_layout()
plt.show()
```
