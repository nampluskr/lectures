```python
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings(action="ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Model

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def gradient(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               retain_graph=True, create_graph=True)[0]

def residual_loss(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    u_x = gradient(u, x)        # du/dx
    u_t = gradient(u, t)        # du/dt
    u_xx = gradient(u_x, x)     # d2u/dxdx
    residual = u_t + u * u_x - (0.01/np.pi) * u_xx
    return torch.mean(residual**2)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 50),  nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1),
        )
    
    def forward(self, x, t):
        inputs = torch.concat([x.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        outputs = self.model(inputs)
        return outputs.squeeze()
```

### Data

```python
x_size, t_size = 256, 100
x_np = np.linspace(-1, 1, x_size)
t_np = np.linspace(0, 0.99, t_size)

x = torch.tensor(x_np).float().to(device)
t = torch.tensor(t_np).float().to(device)

## Initial condition: t = 0, u(x, 0) = -sin(pi*x)
t0 = torch.full_like(x, 0)
x_t0 = x
u_t0 = -torch.sin(np.pi * x)

## Boundary condition: x = -1, u(-1, t) = 0
xmin = torch.full_like(t, -1)
t_xmin = t
u_xmin = torch.full_like(t, 0)

## Boundary condition: x = 1, u(1, t) = 0
xmax = torch.full_like(t, 1)
t_xmax = t
u_xmax = torch.full_like(t, 0)
```

```python
# ## [Random sampling] Collocation points: x in [-1, 1], t in [0, 1]
X_np, T_np = np.meshgrid(x_np, t_np, indexing="ij")     # (x_size, t_size)
x_all = torch.tensor(X_np).float().flatten().to(device)
t_all = torch.tensor(T_np).float().flatten().to(device)

# set_seed(42)
# indices = np.random.permutation(x_size * t_size)
# n_train = 10000
# x_train = torch.tensor(x_all[:n_train]).float().to(device)
# t_train = torch.tensor(t_all[:n_train]).float().to(device)
```

```python
## [Latin hypercube sampling] Collocation points: x in [-1, 1], t in [0, 1]
from pyDOE import lhs

set_seed(42)
n_train = 5000
lb, ub = np.array([x_np.min(), t_np.min()]), np.array([x_np.max(), t_np.max()])
samples = lb + (ub - lb) * lhs(2, n_train)
x_train_np, t_train_np = samples[:, 0], samples[:, 1]

x_train = torch.tensor(x_train_np).float().to(device)
t_train = torch.tensor(t_train_np).float().to(device)
```

### Training

```python
set_seed(42)
model = PINN().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

n_epochs = 10000
for epoch in range(1, n_epochs + 1):
    model.train()
    loss_eqn = residual_loss(model, x_train, t_train)
    loss_t0 = loss_fn(model(x_t0, t0), u_t0)
    loss_xmin = loss_fn(model(xmin, t_xmin), u_xmin)
    loss_xmax = loss_fn(model(xmax, t_xmax), u_xmax)
    
    loss = loss_eqn + loss_t0 + loss_xmin + loss_xmax
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    
    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] loss: {loss.item():.2e} "
              f"(eqn: {loss_eqn.item():.2e} ic: {loss_t0.item():.2e} "
              f"bc1: {loss_xmin.item():.2e} bc2: {loss_xmax.item():.2e})")
```

### Evaluation

```python
import scipy

data = scipy.io.loadmat('burgers_shock.mat')
x_test = data['x'].squeeze()     # (256, 1) -> (256,)
t_test = data['t'].squeeze()     # (100, 1) -> (100, )
U_test = data['usol']            # (256, 100)

X_test, T_test = np.meshgrid(x_test, t_test, indexing="ij")
```

```python
model.eval()
with torch.no_grad():
    x_ = torch.tensor(X_test).flatten().float().to(device)
    t_ = torch.tensor(T_test).flatten().float().to(device)
    pred = model(x_, t_)
    
U_pred = pred.cpu().detach().numpy().reshape(x_size, t_size)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(9, 3))
h1 = ax1.imshow(U_test, interpolation='nearest', cmap='rainbow', 
            extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()], 
            origin='lower', aspect='auto')
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h1, cax=cax1)

h2 = ax2.imshow(U_pred, interpolation='nearest', cmap='rainbow', 
            extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()], 
            origin='lower', aspect='auto')
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h2, cax=cax2)

h3 = ax3.imshow(U_test - U_pred, interpolation='nearest', cmap='rainbow', 
            extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()], 
            origin='lower', aspect='auto')
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h3, cax=cax3)


fig.tight_layout()
plt.show()
```

```python
fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(12, 3))
ax0.plot(x_np, U_pred[:, 0], 'b-', lw=2, label='Exact')
ax0.plot(x_np, U_test[:, 0], 'r--', lw=2, label='Prediction')
ax0.set_title('$t = 0.0s$')

ax1.plot(x_np, U_pred[:, 25], 'b-', lw=2, label='Exact')
ax1.plot(x_np, U_test[:, 25], 'r--', lw=2, label='Prediction')
ax1.set_title('$t = 0.25s$')

ax2.plot(x_np, U_pred[:, 50], 'b-', lw=2, label='Exact')
ax2.plot(x_np, U_test[:, 50], 'r--', lw=2, label='Prediction')
ax2.set_title('$t = 0.50s$')

ax3.plot(x_np, U_pred[:, 75], 'b-', lw=2, label='Exact')
ax3.plot(x_np, U_test[:, 75], 'r--', lw=2, label='Prediction')
ax3.set_title('$t = 0.75s$')

for ax in (ax0, ax1, ax2, ax3):
    ax.legend(fontsize=8)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    
fig.tight_layout()
plt.show()
```
