## Navier-Stokes Equation

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
```

```python
# https://github.com/ComputationalDomain/PINNs/tree/main/Cylinder-Wake

import scipy
data = scipy.io.loadmat("cylinder_wake.mat")

U_star = data["U_star"]             # (5000, 2, 200)
P_star = data["p_star"]             # (5000, 200)
t_star = data["t"]                  # (200, 1)
X_star = data["X_star"]             # (5000, 2)

N = X_star.shape[0]                 # 5000  x_size(50) * y_size(100)
T = t_star.shape[0]                 # 200   t_size

x_test = X_star[:, 0:1]             # (5000, 1)
y_test = X_star[:, 1:2]             # (5000, 1)
p_test = P_star[:, 0:1]             # (5000, 1) initial pressure
u_test = U_star[:, 0:1, 0]          # (5000, 1) initial velocity vx
t_test = np.full_like(x_test, 1)    # (5000, 1)

# Rearrange Data
XX = np.tile(X_star[:, 0:1], (1, T))    # N x T
YY = np.tile(X_star[:, 1:2], (1, T))    # N x T
TT = np.tile(t_star, (1, N)).T          # N x T

UU = U_star[:, 0, :]                    # N x T
VV = U_star[:, 1, :]                    # N x T
PP = P_star                             # N x T

x = XX.flatten()[:, None]               # N*T x 1
y = YY.flatten()[:, None]               # N*T x 1
t = TT.flatten()[:, None]               # N*T x 1

u = UU.flatten()[:, None]               # N*T x 1
v = VV.flatten()[:, None]               # N*T x 1
p = PP.flatten()[:, None]               # N*T x 1

# Training Data
N_train = 5000
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx]
y_train = y[idx]
t_train = t[idx]
u_train = u[idx]
v_train = v[idx]
p_train = p[idx]
```

### Model

```python
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 50),  nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 2)
        )
        
    def forward(self, x, y, t):
        return self.model(torch.hstack([x, y, t]))

    def gradient(self, y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                   create_graph=True, retain_graph=True)[0]

    def residual(self, x, y, t, nu=0.01):
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True

        res = self.forward(x, y, t)
        psi, p = res[:, 0:1], res[:, 1:2]

        u, v = self.gradient(psi, y), -self.gradient(psi, x)
        u_x, u_y, u_t = self.gradient(u, x), self.gradient(u, y), self.gradient(u, t)
        v_x, v_y, v_t = self.gradient(v, x), self.gradient(v, y), self.gradient(v, t)
        u_xx, u_yy = self.gradient(u_x, x), self.gradient(u_y, y)
        v_xx, v_yy = self.gradient(v_x, x), self.gradient(v_y, y)
        p_x, p_y = self.gradient(p, x), self.gradient(p, y)

        f = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        g = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        return torch.mean(f**2), torch.mean(g**2)
    
    def mse(self, x, y, t, u, v, p):
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True

        res = self.forward(x, y, t)
        psi, p_pred = res[:, 0:1], res[:, 1:2]
        u_pred, v_pred = self.gradient(psi, y), -self.gradient(psi, x)
        return torch.mean((u_pred - u)**2), torch.mean((v_pred - v)**2), torch.mean((p_pred - p)**2)
```

### Training

```python
# [ 2000/2000] (lr: 5.99e-04) loss: 6.23e-02 (u: 1.57e-02 v: 3.80e-02 p: 4.14e-03 f: 2.46e-03 g: 2.07e-03)
set_seed(42)
model = PINN().to(device)
optimizer = optim.Adam(model.model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)

x = torch.tensor(x_train).float().to(device)
y = torch.tensor(y_train).float().to(device)
t = torch.tensor(t_train).float().to(device)

u = torch.tensor(u_train).float().to(device)
v = torch.tensor(v_train).float().to(device)
p = torch.tensor(p_train).float().to(device)

zeros = torch.zeros_like(x)

n_epochs = 2000
for epoch in range(1, n_epochs + 1):
    f_loss, g_loss = model.residual(x, y, t)                # pde loss (x_pde, y_pde, t_pde)
    u_loss, v_loss, p_loss = model.mse(x, y, t, u, v, p)    # data loss (x_data, y_data, t_data, u, v, p)
    loss = u_loss + v_loss + p_loss + f_loss + g_loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    
    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] (lr: {scheduler.get_last_lr()[0]:.2e}) "
              f"loss: {loss.item():.2e} "
              f"(u: {u_loss.item():.2e} v: {v_loss.item():.2e} "
              f"p: {p_loss.item():.2e} "
```



