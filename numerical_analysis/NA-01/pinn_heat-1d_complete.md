## Heat Equation-1D
- https://github.com/joseph-nagel/physics-informed-nn/blob/main/notebooks/heat_equation_1d.ipynb

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

### Problem parameters
def u_exact(t, x, alpha, beta):
    return np.sin(beta * x) * np.exp(-alpha * beta**2 * t)

n = 3
t_min, t_max, t_size = 0, 1, 1001
x_min, x_max, x_size = 0, 1, 1001
alpha, beta = 0.05, n*np.pi/x_max
```

### Data
```python
## Data
t = torch.linspace(t_min, t_max, t_size)
x = torch.linspace(x_min, x_max, x_size)
T, X = torch.meshgrid(t, x, indexing="ij")
U = u_exact(T, X, alpha, beta)

n_pde, n_ic, n_bc = 1000, 500, 500

## Domain: u_t - alpha * u_xx = 0
t_pde = torch.rand((n_pde, 1)) * t_max
x_pde = torch.rand((n_pde, 1)) * x_max

## Boundary condition: u(t, 0) = 0
t_bc1 = torch.rand((n_bc, 1)) * t_max
x_bc1 = torch.full_like(t_bc1, x_min)
u_bc1 = torch.full_like(t_bc1, 0)

## Boundary condition: u(t, x_max) = 0
t_bc2 = torch.rand((n_bc, 1)) * t_max
x_bc2 = torch.full_like(t_bc2, x_max)
u_bc2 = torch.full_like(t_bc2, 0)

## Initial condition: u(0, x) = sin(beta * x)
x_ic = torch.rand((n_ic, 1)) * x_max
t_ic = torch.full_like(x_ic, 0)
u_ic = torch.sin(beta * x_ic)
```

```python
def gradient(y, x):
    return torch.autograd.grad(y, x, 
                grad_outputs=torch.ones_like(y), 
                create_graph=True, retain_graph=True)[0]

def residual_loss(model, t, x):
    t.requires_grad = True
    x.requires_grad = True

    u = model(t, x)
    u_t = gradient(u, t)
    u_xx = gradient(gradient(u, x), x)

    residual = u_t - alpha * u_xx
    return torch.mean(residual**2)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 50),  nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, t, x):
        tx = torch.hstack([t, x])
        return self.model(tx)
```

```python
def show_result(t, x, U):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))
    for idx in (0, 50, 100, 150, 200, -1):
        ax1.plot(t, U[idx, :], label=f't={t[idx]:.2f}')
    ax1.set(xlabel='x', ylabel='u(t, x)')
    ax1.set_xlim((t.min(), t.max()))
    ax1.legend(loc="lower right")
    ax1.grid(visible=True, which='both', color='lightgray', linestyle='-')

    img = ax2.imshow(U, cmap='rainbow', aspect='auto', interpolation='bilinear',
                    vmin=np.round(U.min()), vmax=np.round(U.max()),
                    origin='lower', extent=(t.min(), t.max(), x.min(), x.max()))
    ax2.set(xlabel='x', ylabel='t')
    fig.colorbar(img, ax=ax2)
    fig.tight_layout()
    plt.show()
```

```python
## Training
n_epochs = 10000
learning_rate = 1e-3

model = PINN()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

for epoch in range(1, n_epochs + 1):
    model.train()
    loss_pde = residual_loss(model, t_pde, x_pde)
    loss_bc1 = mse_loss(model(t_bc1, x_bc1), u_bc1)
    loss_bc2 = mse_loss(model(t_bc2, x_bc2), u_bc2)
    loss_ic = mse_loss(model(t_ic, x_ic), u_ic)

    loss = loss_pde + loss_bc1 + loss_bc2 + loss_ic
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] (lr: {scheduler.get_last_lr()[0]:.2e}) "
              f"loss: {loss.item():.2e} "
              f"(pde: {loss_pde.item():.2e} ic: {loss_ic.item():.2e} "
              f"bc1: {loss_bc1.item():.2e} bc2: {loss_bc2.item():.2e})")
```

```python
## Evaluation
with torch.no_grad():
    t_test = T.flatten().view(-1, 1)
    x_test = X.flatten().view(-1, 1)

    U_pred = model(t_test, x_test)
    U_pred = U_pred.reshape(x_size, t_size)

show_result(t.numpy(), x.numpy(), U_pred.numpy())
show_result(t.numpy(), x.numpy(), U.numpy())
```
