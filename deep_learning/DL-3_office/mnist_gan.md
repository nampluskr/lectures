```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")
```

```python
import numpy as np
import os
import gzip

def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

data_dir = r"..\data\mnist"
# data_dir = r"..\data\fashion_mnist"

x_train = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
y_train = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
x_test = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
y_test = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")

print(f">> Train images: {x_train.shape}, {x_train.dtype}")
print(f">> Train labels: {y_train.shape}, {y_train.dtype}")
print(f">> Test images:  {x_test.shape}, {x_test.dtype}")
print(f">> Test labels:  {y_test.shape}, {y_test.dtype}")
```

```python
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images.reshape(-1, 28, 28, 1)
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

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

train_loader = DataLoader(Dataset(x_train, y_train, transform=transform), 
                          batch_size=64, shuffle=True)
test_loader = DataLoader(Dataset(x_test, y_test, transform=transform), 
                         batch_size=64, shuffle=True)

x, y = next(iter(train_loader))
print(f">> x: {x.shape}, {x.dtype}, min={x.min()}, max={x.max()}")
print(f">> y: {y.shape}, {y.dtype}, min={y.min()}, max={y.max()}")
```

```python
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 28*28),)

    def forward(self, x):
        x = self.generator(x)
        x = x.view(-1, 1, 28, 28)
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, 1),)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.discriminator(x)
        return torch.sigmoid(x)
```

```python
from torchvision.utils import save_image
from tqdm import tqdm

torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs = 20
learning_rate = 1e-4
latent_dim = 64
step_size = 5
n_outputs = 64

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
loss_fn = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

fixed_noise = torch.randn(n_outputs, latent_dim).to(device)
output_dir = 'd:\\Non_Documents\\lectures\\office\\mnist\\output_gan'
output_images = generator(fixed_noise)
save_image(output_images, os.path.join(output_dir, "fake_image_0.png"), normalize=True)

for epoch in range(1, n_epochs + 1):
    with tqdm(train_loader, leave=False, file=sys.stdout, dynamic_ncols=True, ascii=True) as pbar:
        train_loss_r, train_loss_f, train_loss_g = 0, 0, 0
        for i, (real_images, _) in enumerate(pbar):
            batch_size = len(real_images)
            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)
            noise = torch.randn(batch_size, latent_dim).to(device)
            real_images = real_images.to(device)
            fake_images = generator(noise)

            ## Training Discriminator
            pred_r = discriminator(real_images)
            loss_r = loss_fn(pred_r, real_labels)
            loss_r.backward()

            pred_f = discriminator(fake_images.detach())
            loss_f = loss_fn(pred_f, fake_labels)
            loss_f.backward()

            optimizerD.step()
            optimizerD.zero_grad()

            # Training Generator
            pred_g = discriminator(fake_images)
            loss_g = loss_fn(pred_g, real_labels)
            loss_g.backward()

            optimizerG.step()
            optimizerG.zero_grad()
            
            train_loss_r += loss_r.item()
            train_loss_f += loss_f.item()
            train_loss_g += loss_g.item()

            desc = f"[{epoch:3d}/{n_epochs}] loss_r: {train_loss_r/(i + 1):.2e} " \
                   f"loss_f: {train_loss_f/(i + 1):.2e} loss_g: {train_loss_g/(i + 1):.2e}"

            if i % 10 == 0:
                pbar.set_description(desc)

        if epoch % step_size == 0:
            print(desc)
            output_images = generator(fixed_noise)
            save_image(output_images, os.path.join(output_dir, f"fake_image_{epoch}.png"), normalize=True)
```

```python
# https://github.com/Ksuryateja/DCGAN-MNIST-pytorch/blob/master/gan_mnist.py
class Generator(nn.Module):
    def __init__(self, latent_dim, out_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=1, stride=1, padding=2, bias=False),
        )

    def forward(self, x):
        x = self.generator(x)
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 2, 1, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.discriminator(x)
        return torch.sigmoid(x)
```

```python
from torchvision.utils import save_image
from tqdm import tqdm

torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs = 20
learning_rate = 1e-4
latent_dim = 64
step_size = 5
n_outputs = 64

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
loss_fn = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

fixed_noise = torch.randn(n_outputs, latent_dim, 1, 1).to(device)
output_dir = 'd:\\Non_Documents\\lectures\\office\\mnist\\output_gan'
output_images = generator(fixed_noise)
save_image(output_images, os.path.join(output_dir, "fake_dcgan_0.png"), normalize=True)

for epoch in range(1, n_epochs + 1):
    with tqdm(train_loader, leave=False, file=sys.stdout, dynamic_ncols=True, ascii=True) as pbar:
        train_loss_r, train_loss_f, train_loss_g = 0, 0, 0
        for i, (real_images, _) in enumerate(pbar):
            batch_size = len(real_images)
            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            real_images = real_images.to(device)
            fake_images = generator(noise)

            ## Training Discriminator
            pred_r = discriminator(real_images)
            loss_r = loss_fn(pred_r, real_labels)
            loss_r.backward()

            pred_f = discriminator(fake_images.detach())
            loss_f = loss_fn(pred_f, fake_labels)
            loss_f.backward()
            
            optimizerD.step()
            optimizerD.zero_grad()

            # Training Generator
            pred_g = discriminator(fake_images)
            loss_g = loss_fn(pred_g, real_labels)
            loss_g.backward()

            optimizerG.step()
            optimizerG.zero_grad()
            
            train_loss_r += loss_r.item()
            train_loss_f += loss_f.item()
            train_loss_g += loss_g.item()

            desc = f"[{epoch:3d}/{n_epochs}] loss_r: {train_loss_r/(i + 1):.2e} " \
                   f"loss_f: {train_loss_f/(i + 1):.2e} loss_g: {train_loss_g/(i + 1):.2e}"

            if i % 10 == 0:
                pbar.set_description(desc)

        if epoch % step_size == 0:
            print(desc)
            output_images = generator(fixed_noise)
            save_image(output_images, os.path.join(output_dir, f"fake_dcgan_{epoch}.png"), normalize=True)
```
