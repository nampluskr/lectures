- https://github.com/udemirezen/PINN-1

```python
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from pinn import PINN, Trainer, gradient, to_tensor

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
def u_exact(x, t):
    return np.sin(np.pi * x)*np.exp(-np.pi**2 * t)

def residual_loss(model, inputs):
    x, t = inputs
    x.requires_grad = True
    t.requires_grad = True
    u = model([x, t])
    u_t = gradient(u, t)
    u_x = gradient(u, x)
    u_xx = gradient(u_x, x)
    residual = u_t - u_xx
    return torch.mean(residual**2)

loss_functions = {}
loss_functions["residual"] = residual_loss
```

```python
x_np = np.random.rand(1000)
t_np = np.random.rand(1000)
u_np = u_exact(x_np, t_np)

x, t, u = to_tensor(x_np), to_tensor(t_np), to_tensor(u_np)

targets = {}
targets["data"] = [x, t], u
inputs = [x, t]
```

```python
# Hyperparameters
layers = [2, 20, 1]
learning_rate = 1e-3
n_epochs = 10000

model = PINN(layers_dim=layers, activation="tanh").to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.985)

solver = Trainer(model, optimizer, loss_functions, targets)
solver.fit(inputs, n_epochs, scheduler=scheduler, update_step=20)
solver.show_history()
```

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
solver.fit(inputs, n_epochs, scheduler=scheduler, update_step=20)
solver.show_history()
```
