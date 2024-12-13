```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")
```

```python
import os
import numpy as np
import pickle

def unpickle(filename):
    # tar -zxvf cifar-10-python.tar.gz
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    x = np.array(data[b'data']).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(data[b'labels'])
    return x, y


def load_cifar10(data_dir):
    batch_files = [os.path.join(data_dir, f"data_batch_{i+1}") for i in range(5)]
    test_file = os.path.join(data_dir, "test_batch")

    images, labels = [], []
    for filename in batch_files:
        x, y = unpickle(filename)
        images.append(x)
        labels.append(y)

    x_train = np.concatenate(images, axis=0)
    y_train = np.concatenate(labels, axis=0)

    x_test, y_test = unpickle(test_file)
    return (x_train, y_train), (x_test, y_test)

data_dir = r"..\data\cifar10\cifar-10-batches-py"
(x_train, y_train), (x_test, y_test) = load_cifar10(data_dir)

print(f">> Train images: {x_train.shape}, {x_train.dtype}")
print(f">> Train labels: {y_train.shape}, {y_train.dtype}")
print(f">> Test images:  {x_test.shape}, {x_test.dtype}")
print(f">> Test labels:  {y_test.shape}, {y_test.dtype}")
```

```python
import torchvision.transforms as transforms

class CIFAR10(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).long()
        return image, label

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomVerticalFlip(0.3),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

train_dataset = CIFAR10(x_train, y_train, transform=transform_train)
test_dataset = CIFAR10(x_test, y_test, transform=transform_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))
print(f">> x: {x.shape}, {x.dtype}, min={x.min()}, max={x.max()}")
print(f">> y: {y.shape}, {y.dtype}, min={y.min()}, max={y.max()}")
```

```python
## Model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        x = self.conv_block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv_block1 = ConvBlock(3, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)
        self.fc1 = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc2 = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(-1, 128 * 4 * 4)
        mu, logvar = self.fc1(x), self.fc2(x)
        return mu, logvar


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.deconv_block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = DeconvBlock(128, 64)
        self.deconv2 = DeconvBlock(64, 32)
        self.deconv3 = nn.ConvTranspose2d(32, output_channels,
                            kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampling(mu, logvar)
        x = self.decoder(z)
        x = torch.sigmoid(x)
        return x, mu, logvar

def binary_accuracy(x_pred, x_true):
    return torch.eq(x_pred.round(), x_true.round()).float().mean()

def bce_kld_loss(x_pred, x, mu, logvar):
    bce = nn.BCELoss(reduction='sum')(x_pred, x)
    kdl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kdl
```

```python
from common.trainer import Trainer

class VAETrainer(Trainer):
    def train_step(self, x, y):
        x = x.to(self.device)
        x_pred, mu, logvar = self.model(x)
        loss = self.loss_fn(x_pred, x, mu, logvar)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        res = {"loss": loss.item()}
        res.update({name: func(x_pred, x).item() 
                    for name, func in self.metrics.items() if name != "loss"})
        return res

    @torch.no_grad()
    def test_step(self, x, y):
        x = x.to(self.device)
        x_pred, mu, logvar = self.model(x)
        loss = self.loss_fn(x_pred, x, mu, logvar)

        res = {"loss": loss.item()}
        res.update({name: func(x_pred, x).item() 
                    for name, func in self.metrics.items() if name != "loss"})
        return res
```

```python
from common.trainer import EarlyStopping

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 64
n_epochs = 100
learning_rate = 1e-3

encoder = Encoder(latent_dim)
decoder = Decoder(latent_dim)
model = VAE(encoder, decoder).to(device)
loss_fn = bce_kld_loss
metrics = {"bce": nn.BCELoss(), "acc": binary_accuracy}
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

vae = VAETrainer(model, optimizer, loss_fn, metrics=metrics)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
stopper = EarlyStopping(patience=3, min_delta=10)

history = vae.fit(train_loader, n_epochs, valid_loader=test_loader,
                 scheduler=scheduler, stopper=stopper, step_size=1)
```
