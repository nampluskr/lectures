## Harmonic Ocsillator

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
## Equation parameters:
def u_exact(t, d, w0):
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d/w)
    A = 1 / (2*np.cos(phi))
    return 2*A*np.cos(phi + w*t)*np.exp(-d*t)

d, w0 = 2, 20           # reference value w0 = 20
mu, k = 2*d, w0**2

t_min, t_max, t_size = 0, 1, 101
t = np.linspace(t_min, t_max, t_size)

def show_result(t, u=None):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t, u_exact(t, d, w0), 'k:', label="Exact")
    if u is not None:
        ax.plot(t, u, 'r', lw=1, label="Prediction")
    ax.legend(loc="upper right")
    ax.grid(color='k', ls=':', lw=.5)
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    fig.tight_layout()
    plt.show()
    
show_result(t)
```

### Data

```python
def to_tensor(x):
    return torch.tensor(x).float().view(-1, 1).to(device)

n_pde = 1001

## Domain: u_tt + nu * u_t + k * u
t_pde = np.random.rand(n_pde) * (t_max - t_min) + t_min
t_pde = to_tensor(t_pde)

## Initial contidion: u(0) = 1

t_ic = to_tensor(0)
u_ic = to_tensor(1)
print(t_pde.shape, t_ic.shape, u_ic.shape)
```

```python
def gradient(y, x):
    return torch.autograd.grad(y, x, 
                grad_outputs=torch.ones_like(y), 
                create_graph=True, retain_graph=True)[0]

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 50),  nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, t):
        return self.model(t)
    
    def residual(self, t, mu, k):
        t.requires_grad = True

        u = self.forward(t)
        u_t = gradient(u, t)
        u_tt = gradient(u_t, t)

        residual = u_tt + mu * u_t + k * u
        return torch.mean(residual**2)
    
    def initial(self, t):
        t.requires_grad = True
        u = self.forward(t)
        u_t = gradient(u, t)
        return torch.mean(u_t**2)
    
    def mse(self, t, u):
        u_pred = self.forward(t)
        return torch.mean((u_pred - u)**2)
```

### Training

```python
## Training
n_epochs = 20000
learning_rate = 1e-3

model = PINN().to(device)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.985)

for epoch in range(1, n_epochs + 1):
    model.train()
    loss_pde = model.residual(t_pde, mu, k)
    loss_ic_du = model.initial(t_ic)
    loss_ic_u = model.mse(t_ic, u_ic)

    loss = loss_pde * 1e-4 + loss_ic_du * 1e-2 + loss_ic_u
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] (lr: {scheduler.get_last_lr()[0]:.2e}) "
              f"loss: {loss.item():.2e} (pde: {loss_pde.item():.2e} "
              f"ic_du: {loss_ic_du.item():.2e} ic_u: {loss_ic_u.item():.2e})")
```

### Evaluation

```python
## Evaluation
with torch.no_grad():
    t_test = to_tensor(t)
    u_pred = model(t_test)
    
u_pred = u_pred.cpu().detach().numpy().flatten()
show_result(t, u_pred)
```
