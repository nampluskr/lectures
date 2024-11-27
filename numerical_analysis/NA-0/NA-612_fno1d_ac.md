### Fourier Neural Net 1D - Allen-Cahn equation

- Code:
  - https://github.com/hl4220/Extended-Physics-Informed-Neural-Operator/blob/main/Anti-derivative/PI_deepOnet_two_step_training.ipynb

- Data:
  - https://github.com/camlab-ethz/AI_Science_Engineering/blob/main/datasets/AC_data_input.npy
  - https://github.com/camlab-ethz/AI_Science_Engineering/blob/main/datasets/AC_data_output.npy
 
```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # current gpu seed
torch.cuda.manual_seed_all(seed) # All gpu seed
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = False  # True로 하면 gpu에 적합한 알고리즘을 선택함.
```

### Model

```python
class SpectralConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        # (batch_size, in_channels, x), (in_channels, out_channels, x) -> (batch_size, out_channels, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # x: [batch_size, in_channels, number of grid points]
        batch_size, n_points = x.shape[0], x.shape[-1]

        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batch_size, self.out_channels, n_points // 2 + 1).cfloat().to(x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=n_points)
        return x
```

```python
class FNO1D(nn.Module):
    def __init__(self, modes, width):
        super().__init__()
        self.padding = 1  # pad the domain if input is non-periodic
        self.input_layer = nn.Linear(2, width)  # input channel is 2: (u0(x), x) --> GRID IS INCLUDED!

        self.spect1 = SpectralConv1D(width, width, modes)
        self.spect2 = SpectralConv1D(width, width, modes)
        self.spect3 = SpectralConv1D(width, width, modes)
        self.conv1 = nn.Conv1d(width, width, kernel_size=1)
        self.conv2 = nn.Conv1d(width, width, kernel_size=1)
        self.conv3 = nn.Conv1d(width, width, kernel_size=1)
        self.activation = nn.Tanh()

        self.output_layer = nn.Sequential(
            nn.Linear(width, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    # def fourier_layer(self, x, spect_layer, conv_layer):
    #     return self.activation(spect_layer(x) + conv_layer(x))

    # def linear_layer(self, x, linear_transformation):
    #     return self.activation(linear_transformation(x))

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)

        x = self.input_layer(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0, self.padding])   # pad the domain if input is non-periodic

        x = self.activation(self.spect1(x) + self.conv1(x))
        x = self.activation(self.spect2(x) + self.conv2(x))
        x = self.activation(self.spect3(x) + self.conv3(x))

        # x = self.fourier_layer(x, self.spect1, self.conv0)
        # x = self.fourier_layer(x, self.spect2, self.conv1)
        # x = self.fourier_layer(x, self.spect3, self.conv2)

        # x = x[..., :-self.padding]        # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)
        return x
```

### Data

```python
## Data
from torch.utils.data import TensorDataset, DataLoader

n_train = 100
x_data_np = np.load("AC_data_input.npy")
y_data_np = np.load("AC_data_output.npy")
print(">> Data: ", x_data_np.shape, y_data_np.shape)

## torch tensors
x_data = torch.tensor(x_data_np).float()
y_data = torch.tensor(y_data_np).float()

temp = torch.clone(x_data[..., 0])
x_data[..., 0] = x_data[..., 1]
x_data[..., 1] = temp

x_train = x_data[:n_train]
y_train = y_data[:n_train]
x_test = x_data[n_train:]
y_test = y_data[n_train:]

batch_size = 32
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))
print(">> Batch:", x.shape, y.shape)
```

```python
idx = 32
mesh = x_train[idx, :, 1]

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(mesh, x_train[idx, :, 0], label = "Input: function")
ax.plot(mesh, y_train[idx], label = "Output")
ax.grid(True, which="both", ls=":")
ax.legend()
fig.tight_layout()
plt.show()
```

### Training

```python
## Training
learning_rate = 1e-3
n_epochs = 200

modes, width = 16, 64
model = FNO1D(modes, width).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred.squeeze(), y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    scheduler.step()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred.squeeze(), y)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:3d}/{n_epochs}] train_loss: {train_loss:.2e} test_loss: {test_loss:.2e}")
```

### Evaluation

```python
## Evaluation
with torch.no_grad():
    y_pred = model(x_test.to(device))

x_test_np = x_test.cpu().detach().numpy()
y_test_np = y_test.cpu().detach().numpy().squeeze()
y_pred_np = y_pred.cpu().detach().numpy().squeeze()
print(x_test_np.shape, y_test_np.shape, y_pred_np.shape)

idx = 59
mesh = x_test_np[idx, :, 1]

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(mesh, x_test_np[idx, :, 0], 'g', label = "Input")
ax.plot(mesh, y_test_np[idx], 'k:', label = "Output")
ax.plot(mesh, y_pred_np[idx], 'k', label = "Prediction")
ax.grid(True, which="both", ls=":")
ax.legend()
fig.tight_layout()
plt.show()
```
