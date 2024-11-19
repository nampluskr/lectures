## Operator Learing - Fourier Neural Operator

- Reference
  - https://github.com/neuraloperator/neuraloperator/blob/af93f781d5e013f8ba5c52baa547f2ada304ffb0/fourier_1d.py
  - https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/neural_operators/simple_FNO_in_JAX.ipynb

- Data
  - [615M] https://ssd.mathworks.com/supportfiles/nnet/data/burgers1d/burgers_data_R10.mat
 
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import scipy

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Model

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, activation=nn.ReLU()):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1),
            activation,
            nn.Conv1d(mid_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)

class FFTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes, activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        weights = torch.rand(in_channels, out_channels, modes).cfloat() 
        scale = 1 / in_channels / out_channels
        self.weights = nn.Parameter(weights * scale)
        self.conv = ConvBlock(out_channels, out_channels, out_channels, activation=nn.GELU())
        self.skip = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(x.shape[0], self.out_channels, x.shape[-1]//2 + 1).cfloat().to(x.device)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights)
        x1 = torch.fft.irfft(out_ft, n=x.shape[-1])
        x1 = self.conv(x1)
        x2 = self.skip(x)
        return (x1 + x2) if self.activation is None else self.activation(x1 + x2)

class FNO1d(nn.Module):
    def __init__(self, modes, width, activation=nn.ReLU()):
        super().__init__()
        self.padding = 8 # pad the domain if input is non-periodic

        self.p_layer = nn.Linear(2, width)
        self.fft_block0 = FFTBlock(width, width, modes, activation=activation)
        self.fft_block1 = FFTBlock(width, width, modes, activation=activation)
        self.fft_block2 = FFTBlock(width, width, modes, activation=activation)
        self.fft_block3 = FFTBlock(width, width, modes, activation=None)
        self.q_layer = ConvBlock(width, 1, width*2, activation=nn.ReLU())

    def forward(self, x):
        grid = torch.linspace(0, 1, x.shape[1]).repeat([x.shape[0], 1]).unsqueeze(-1).to(x.device)
        x = torch.cat((x, grid), dim=-1)        # (batch_size, n_points, 2)
        x = self.p_layer(x).permute(0, 2, 1)    # (batch_sdize, width, n_points)
        # x = F.pad(x, [0, self.padding])       # pad the domain if input is non-periodic

        x = self.fft_block0(x)
        x = self.fft_block1(x)
        x = self.fft_block2(x)
        x = self.fft_block3(x)

        # x = x[..., :-self.padding]            # pad the domain if input is non-periodic
        x = self.q_layer(x).permute(0, 2, 1)
        return x
```

### Data

```python
from torch.utils.data import DataLoader, TensorDataset

## Data
n_train = 1000
n_test = 100

sub_sampling = 2**3          # subsampling rate
batch_size = 64

data = scipy.io.loadmat("burgers_data_R10.mat")
x_data_np = data["a"]   # a(x): initial condition
y_data_np = data["u"]   # u(x): PDE solution

x_data = torch.tensor(x_data_np).float()[:, ::sub_sampling]
y_data = torch.tensor(y_data_np).float()[:, ::sub_sampling]

x_train = x_data[:n_train].unsqueeze(-1)
y_train = y_data[:n_train].unsqueeze(-1)
x_test = x_data[-n_test:].unsqueeze(-1)
y_test = y_data[-n_test:].unsqueeze(-1)
print(x_train.shape, x_test.shape)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
```

### Train

```python
n_epochs = 200
learning_rate = 1e-4
iterations = n_epochs*(n_train // batch_size)
modes = 16
width = 128

model = FNO1d(modes, width, activation=nn.ReLU()).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
# loss_fn = nn.HuberLoss(reduction='sum', delta=0.7)
loss_fn = nn.MSELoss(reduction='sum')

for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()

        train_loss += loss.item()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_loss += loss_fn(out, y).item()

    train_loss /= len(train_loader)
    test_loss /= len(test_loader)

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch}/{n_epochs}] train_loss: {train_loss:.2e} test_loss: {test_loss:.2e}")
```

```
[20/200] train_loss: 1.13e+01 test_loss: 9.51e+00
[40/200] train_loss: 3.14e+00 test_loss: 3.62e+00
[60/200] train_loss: 1.56e+00 test_loss: 2.09e+00
[80/200] train_loss: 1.03e+00 test_loss: 1.36e+00
[100/200] train_loss: 7.80e-01 test_loss: 9.59e-01
[120/200] train_loss: 6.44e-01 test_loss: 7.29e-01
[140/200] train_loss: 6.83e-01 test_loss: 9.51e-01
[160/200] train_loss: 6.02e-01 test_loss: 9.55e-01
[180/200] train_loss: 3.87e-01 test_loss: 5.15e-01
[200/200] train_loss: 5.36e-01 test_loss: 4.85e-01

[10/100] train_loss: 1.15e+00 test_loss: 1.25e+00
[20/100] train_loss: 3.39e-01 test_loss: 3.67e-01
[30/100] train_loss: 2.51e-01 test_loss: 5.77e-01
[40/100] train_loss: 1.68e-01 test_loss: 5.18e-01
[50/100] train_loss: 1.61e-01 test_loss: 1.50e-01
[60/100] train_loss: 1.00e-01 test_loss: 1.55e-01
[70/100] train_loss: 9.12e-02 test_loss: 1.07e-01
[80/100] train_loss: 8.70e-02 test_loss: 1.05e-01
[90/100] train_loss: 8.14e-02 test_loss: 9.51e-02
[100/100] train_loss: 8.06e-02 test_loss: 9.65e-02
```
