## DeepONet Tutorial in pytorch

- Code: DeepONet
  - https://github.com/GideonIlung/DeepONet/blob/main/src/model.py
  - https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/neural_operators/simple_deepOnet_in_JAX.ipynb

- Data: ETH Zurich's course on "Deep Learning in Scienfitic Computing"
  - https://github.com/mroberto166/CAMLab-DLSCTutorials/raw/main/antiderivative_aligned_train.npz
  - https://github.com/mroberto166/CAMLab-DLSCTutorials/raw/main/antiderivative_aligned_test.npz

```python
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
```

```python
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
## Data
data_train = np.load("antiderivative_aligned_train.npz", allow_pickle=True)
branch_inputs_train = data_train["X"][0]    # (n_samples, n_points)
trunk_inputs_train = data_train["X"][1]     # (n_points, 1)
outputs_train = data_train["y"]             # (n_samples, n_points)

data_test = np.load("antiderivative_aligned_test.npz", allow_pickle=True)
branch_inputs_test = data_test["X"][0]      # (n_samples, n_points)
trunk_inputs_test = data_test["X"][1]       # (n_points, 1)
outputs_test = data_test["y"]               # (n_samples, n_points)

print(">> Train Data:", branch_inputs_train.shape, trunk_inputs_train.shape, outputs_train.shape)
print(">> Test Data: ", branch_inputs_test.shape, trunk_inputs_test.shape, outputs_test.shape)
```

```python
idx = 2
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(trunk_inputs_train.flatten(), branch_inputs_train[idx, :], label="input")
ax.plot(trunk_inputs_train.flatten(), outputs_train[idx, :], label="antiderivative")
ax.legend(loc="upper right")
ax.grid(color='k', ls=':', lw=1)
ax.set_xlabel("t")
ax.set_ylabel("u(t)")
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
## Model
n_points = 100
hidden_size, interact_size = 50, 50
n_epochs = 10000
learning_rate = 1e-4

branch_layers = [n_points, hidden_size, hidden_size, interact_size]
branch_net = MLP(branch_layers, activation=nn.ReLU())

trunk_layers = [1, hidden_size, hidden_size, interact_size]
trunk_net = MLP(trunk_layers, activation=nn.ReLU(), final_activation=nn.ReLU())

model = DeepONet1D(branch_net, trunk_net).to(device)

## Train
x_branch = torch.tensor(branch_inputs_train).float().to(device) # (n_samples, n_points)
x_trunk = torch.tensor(trunk_inputs_train).float().to(device)   # (n_points, 1)
out_train = torch.tensor(outputs_train).float().to(device)      # (n_samples, n_points)

loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(1, n_epochs + 1):
    out = model(x_branch, x_trunk)
    loss = loss_fn(out, out_train)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch}/{n_epochs}] loss: {loss.item():.2e}")
```

```python
## Test
x_branch_test = torch.tensor(branch_inputs_test).float().to(device) # (n_samples, n_points)
x_trunk_test = torch.tensor(trunk_inputs_test).float().to(device)   # (n_points, 1)
out_test = torch.tensor(outputs_test).float().to(device)            # (n_samples, n_points)

with torch.no_grad():
    out = model(x_branch_test, x_trunk_test)

out_np = out.cpu().detach().numpy()

idx = 30
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(trunk_inputs_test.flatten(), branch_inputs_test[idx, :], 'k', label="Input")
ax.plot(trunk_inputs_test.flatten(), outputs_test[idx, :], 'k:', label="Exact")
ax.plot(trunk_inputs_test.flatten(), out_np[idx, :], 'r', lw=1, label="Prediction")
ax.legend(loc="upper right")
ax.grid(color='k', ls=':', lw=1)
ax.set_xlabel("t")
ax.set_ylabel("u(t)")
fig.tight_layout()
plt.show()
```
