## Navier-Stokes Equation: Flow around Cylinder

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

```python
# https://github.com/ComputationalDomain/PINNs/blob/main/Cylinder-Wake/NS_PINNS.py

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def gradient(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               retain_graph=True, create_graph=True)[0]

def residual_loss(model, x, y, t):
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True
    output = model(x, y, t)
    u, v, p = output.T
    
    u_x, u_y, u_t = gradient(u, x), gradient(u, y), gradient(u, t)
    u_xx, u_yy = gradient(u_x, x), gradient(u_y, y)
    
    v_x, v_y, v_t = gradient(v, x), gradient(v, y), gradient(v, t)
    v_xx, v_yy = gradient(v_x, x), gradient(v_y, y)
    
    p_x, p_y = gradient(p, x), gradient(p, y)
    
    nu = 0.01
    residual_x = (u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy))
    residual_y = (v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy))
    return torch.mean(residual_x**2) + torch.mean(residual_y**2)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 20),  nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 3),
        )
    
    def forward(self, x, y, t):
        inputs = torch.concat([x.unsqueeze(-1), y.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        outputs = self.model(inputs)
        return outputs
```

```python
import scipy
data = scipy.io.loadmat("cylinder_wake.mat")

N, T = 5000, 200
x_np = data["X_star"][:, 0]         # (N,)
y_np = data["X_star"][:, 1]         # (N,)
t_np = data["t"].reshape(-1)        # (T,)

X_np = np.tile(x_np.reshape(-1, 1), [1, T])
Y_np = np.tile(y_np.reshape(-1, 1), [1, T])
T_np = np.tile(t_np.reshape(-1, 1), [1, N]).T

U_np = data["U_star"][:, 0, :]      # (N, T)
V_np = data["U_star"][:, 1, :]      # (N, T)
P_np = data["p_star"]               # (N, T)

print(x_np.shape, y_np.shape, t_np.shape)
print(X_np.shape, Y_np.shape, T_np.shape)
print(U_np.shape, V_np.shape, P_np.shape)
```

```python
n_train = 20000
set_seed(42)
indices = np.random.choice(N*T, n_train, replace=False)

x_train = torch.tensor(X_np.flatten()[indices]).float().to(device)
y_train = torch.tensor(Y_np.flatten()[indices]).float().to(device)
t_train = torch.tensor(T_np.flatten()[indices]).float().to(device)

u_train = torch.tensor(U_np.flatten()[indices]).float().to(device)
v_train = torch.tensor(V_np.flatten()[indices]).float().to(device)
p_train = torch.tensor(P_np.flatten()[indices]).float().to(device)

## (n_train, 3)
s_train = torch.concat([u_train.view(-1, 1), v_train.view(-1, 1), p_train.view(-1, 1)], dim=-1)
```

### Training

```python
n_epochs = 10000
learning_rate = 1e-3

model = PINN().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.985)

for epoch in range(1, n_epochs + 1):
    model.train()
    
    loss_eqn = residual_loss(model, x_train, y_train, t_train)
    loss_data = loss_fn(model(x_train, y_train, u_train), s_train)
    
    loss = loss_eqn * 1e-3 + loss_data
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    
    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] (lr: {scheduler.get_last_lr()[0]:.2e}) "
              f"loss: {loss.item():.2e} "
              f"(eqn: {loss_eqn.item():.2e} data: {loss_data.item():.2e})")
```

```python
n_epochs = 10000
learning_rate = 1e-4

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.985)

for epoch in range(1, n_epochs + 1):
    model.train()
    
    loss_eqn = residual_loss(model, x_train, y_train, t_train)
    loss_data = loss_fn(model(x_train, y_train, u_train), s_train)
    
    loss = loss_eqn * 1e-3 + loss_data
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    
    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] (lr: {scheduler.get_last_lr()[0]:.2e}) "
              f"loss: {loss.item():.2e} "
              f"(eqn: {loss_eqn.item():.2e} data: {loss_data.item():.2e})")
```

### Evaluation

```python
x_test = torch.tensor(X_np.flatten()).float().to(device)
y_test = torch.tensor(Y_np.flatten()).float().to(device)
t_test = torch.tensor(T_np.flatten()).float().to(device)

with torch.no_grad():
    pred = model(x_test, y_test, t_test)
    
u_pred, v_pred, p_pred = pred.cpu().detach().numpy().T
u_pred = u_pred.reshape(50, 100, T)
v_pred = v_pred.reshape(50, 100, T)
p_pred = p_pred.reshape(50, 100, T)

u_data = U_np.reshape(50, 100, T)
v_data = V_np.reshape(50, 100, T)
p_data = P_np.reshape(50, 100, T)


time = 0
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(9, 3))
h1 = ax1.imshow(u_pred[..., time], interpolation='nearest', cmap='rainbow', 
                # extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()], 
                origin='lower', aspect='auto')
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h1, cax=cax1)

h2 = ax2.imshow(v_pred[..., time], interpolation='nearest', cmap='rainbow', 
                # extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()], 
                origin='lower', aspect='auto')
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h2, cax=cax2)

h3 = ax3.imshow(p_pred[..., time], interpolation='nearest', cmap='rainbow', 
                # extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()], 
                origin='lower', aspect='auto')
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h3, cax=cax3)

fig.tight_layout()
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(9, 3))
h1 = ax1.imshow(u_data[..., time], interpolation='nearest', cmap='rainbow', 
                # extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()], 
                origin='lower', aspect='auto')
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h1, cax=cax1)

h2 = ax2.imshow(v_data[..., time], interpolation='nearest', cmap='rainbow', 
                # extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()], 
                origin='lower', aspect='auto')
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h2, cax=cax2)

h3 = ax3.imshow(p_data[..., time], interpolation='nearest', cmap='rainbow', 
                # extent=[t_test.min(), t_test.max(), x_test.min(), x_test.max()], 
                origin='lower', aspect='auto')
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h3, cax=cax3)

fig.tight_layout()
plt.show()
```
