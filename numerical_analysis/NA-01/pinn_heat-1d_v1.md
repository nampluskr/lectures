### Initial condition - Divergent

### 1D Heat Equation:

$$\frac{\partial u}{\partial t} = \alpha\frac{\partial^2 u}{\partial x^2},\quad t\in [0, T],\quad x\in [0, L]$$

- Initial condition:
$$u(0, x) = \sin\left(\frac{n\pi x}{L}\right),\quad x\in [0, L]$$

- Boundary conditions:
$$u(t, 0) = u(t, L) = 0,\quad t\in [0, T]$$


```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def gradient(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                        create_graph=True, retain_graph=True)[0]

def to_tensor(x):
    return torch.tensor(x).float().to(device)

def to_array(x):
    return x.cpu().detach().numpy()

def diff(y, xs):
    """
        u_pred = model([t, x, y])
        diff(u_pred, [x, x])    # dudxx
        diff(u_pred, [x, y, t]) # dudxyt
    """
    grad = y
    for x in xs:
        grad = gradient(grad, x)
    return grad
```

```python
# https://github.com/joseph-nagel/physics-informed-nn/blob/main/notebooks/heat_equation_1d.ipynb
def u_exact(t, x, alpha, beta):
    return np.sin(beta * x) * np.exp(-alpha * beta**2 * t)

T, L, n = 1, 1, 3
alpha, beta = 0.07, n * np.pi / L

t_min, t_max, t_size = 0, T, 1001
x_min, x_max, x_size = 0, L, 1001

t_np = np.linspace(t_min, t_max, t_size)
x_np = np.linspace(x_min, x_max, x_size)
T_np, X_np = np.meshgrid(t_np, x_np, indexing="ij")
U_np = u_exact(T_np, X_np, alpha=alpha, beta=beta)
print(t_np.shape, x_np.shape)
print(T_np.shape, X_np.shape, U_np.shape)
```

```python
def show_result(x, t, U):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))
    for idx in (0, 50, 100, 150, 200, -1):
        ax1.plot(x, U[idx,:], label=f't={t[idx]}')
    ax1.set(xlabel='x', ylabel='u(t, x)')
    ax1.set_xlim((x.min(), x.max()))
    ax1.legend(loc="lower right")
    ax1.grid(visible=True, which='both', color='lightgray', linestyle='-')

    img = ax2.imshow(U.T, cmap='rainbow', aspect='auto', interpolation='bilinear',
                    vmin=np.round(U.min()), vmax=np.round(U.max()),
                    origin='lower', extent=(t.min(), t.max(), x.min(), x.max()))
    ax2.set(xlabel='t', ylabel='x')
    fig.colorbar(img, ax=ax2)
    fig.tight_layout()
    plt.show()
    
show_result(t_np, x_np, U_np)
```

```python
def residual_loss(model, t, x, alpha=alpha):
    t.requires_grad = True
    x.requires_grad = True
    u = model(t, x)
    u_t = gradient(u, t)
    u_x = gradient(u, x)
    u_xx = gradient(u_x, x)
    residual = u_t - alpha * u_xx
    return torch.mean(residual**2)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),   nn.Tanh(),
            nn.Linear(32, 32),  nn.Tanh(),
            nn.Linear(32, 32),  nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t, x):
        # x = torch.concat([t.unsqueeze(-1), x.unsqueeze(-1)], dim=-1)
        x = torch.hstack([t.view(-1, 1), x.view(-1, 1)])
        return self.model(x)
```

```python
num_pde, num_bc, num_ic = 5000, 500, 500

t_train = torch.rand(num_pde).to(device) * t_max
x_train = torch.rand(num_pde).to(device) * x_max
bc_t = torch.rand(num_bc).to(device) * t_max
ic_x = torch.rand(num_ic).to(device) * x_max

t0 = torch.full_like(ic_x, 0)
x_t0 = ic_x
u_t0 = torch.sin(beta * ic_x)

t_xmin = bc_t
xmin = torch.full_like(bc_t, 0)
u_xmin = torch.full_like(bc_t, 0)

t_xmax = bc_t
xmax = torch.full_like(bc_t, x_max)
u_xmax = torch.full_like(bc_t, 0)
```

```python
# from pyDOE import lhs

# t, x = to_tensor(t_np), to_tensor(x_np)

# ## Initial condition: u(0, x) = sin(beta * x), beta = n*pi/L
# t0 = torch.full_like(x, 0)
# x_t0 = x
# u_t0 = torch.sin(beta * x)
# # u_t0 = torch.full_like(x, 1)

# ## Boundary conditoin: u(t, 0) = 0
# t_xmin = t
# xmin = torch.full_like(t, 0)
# u_xmin = torch.full_like(t, 0)

# ## Boundary condition: u(t, L) = 0
# t_xmax = t
# xmax = torch.full_like(t, L)
# u_xmax = torch.full_like(t, 0)

# ## Domain (collocation points)
# set_seed(42)
# n_points = 10000
# lb, ub = np.array([t_min, x_min]), np.array([t_max, x_max])
# points = lb + (ub - lb) * lhs(2, n_points)
# t_train = to_tensor(points[:, 0])
# x_train = to_tensor(points[:, 1])
```

```python
fig, ax1 = plt.subplots(figsize=(5, 3))
h1 = ax1.imshow(U_np.T, 
                origin='lower', aspect='auto',
                interpolation='bilinear', cmap='rainbow', 
                extent=[t_min, t_max, x_min, x_max],)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h1, cax=cax1)

ax1.plot(to_array(t_train), to_array(x_train), 'ko', ms=1)
ax1.plot(to_array(t0), to_array(x_t0), 'rx', ms=5)
ax1.plot(to_array(t_xmin), to_array(xmin), 'bx', ms=5)
ax1.plot(to_array(t_xmax), to_array(xmax), 'gx', ms=5)
ax1.set_xlabel("t")
ax1.set_ylabel("x")
fig.tight_layout()
plt.show()
```

```python
set_seed(42)
model = PINN().to(device)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

n_epochs = 10000
for epoch in range(1, n_epochs + 1):
    model.train()
    loss_eqn = residual_loss(model, t_train, x_train)
    loss_ic = mse_loss(model(t0, x_t0), u_t0)
    loss_bc1 = mse_loss(model(t_xmin, xmin), u_xmin)
    loss_bc2 = mse_loss(model(t_xmax, xmax), u_xmax)
    loss = loss_eqn + loss_ic + loss_bc1 + loss_bc2

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # scheduler.step()
    
    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] (lr: {scheduler.get_last_lr()[0]:.2e}) "
              f"loss: {loss.item():.2e} eqn: {loss_eqn.item():.2e} "
              f"ic: {loss_ic.item():.2e} "
              f"bc1: {loss_bc1.item():.2e} bc2: {loss_bc2.item():.2e}")
```

```python
T, X = to_tensor(T_np), to_tensor(X_np)
t_test, x_test = T.flatten(), X.flatten()

with torch.no_grad():
    U_pred = model(t_test, x_test)
    
U_pred = to_array(U_pred).reshape(t_size, x_size)

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(5, 6))
h1 = ax1.imshow(U_np.T, 
                origin='lower', aspect='auto',
                interpolation='bilinear', cmap='rainbow', 
                extent=[t_min, t_max, x_min, x_max],)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h1, cax=cax1)

h2 = ax2.imshow(U_pred.T, 
                origin='lower', aspect='auto',
                interpolation='bilinear', cmap='rainbow', 
                extent=[t_min, t_max, x_min, x_max],)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="3%", pad=0.1)
fig.colorbar(h2, cax=cax2)

for ax in (ax1, ax2):
    ax.set_xlabel("t")
    ax.set_ylabel("x")
fig.tight_layout()
plt.show()
```
