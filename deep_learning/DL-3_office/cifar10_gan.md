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
    # transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(0.3),
    # transforms.RandomVerticalFlip(0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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
class Generator(nn.Module):
    def __init__(self, latent_dim):
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
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.generator(x)
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.discriminator(x)
        return torch.sigmoid(x)
```

```python
from torchvision.utils import save_image
from tqdm import tqdm

class GanTrainer:
    def __init__(self, generator, discriminator, optimizerG, optimizerD, n_outputs=64):
        self.latent_dim = generator.latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.loss_fn = nn.BCELoss()
        self.device = next(generator.parameters()).device
        self.n_outputs = n_outputs
        
    def fit(self, train_loader, n_epochs, output_dir=None, step_size=5):
        fixed_noise = torch.randn(self.n_outputs, self.latent_dim, 1, 1).to(self.device)
        if output_dir is not None:
            output_images = self.generator(fixed_noise)
            save_image(output_images, os.path.join(output_dir, "fake_image_0.png"), normalize=True)

        for epoch in range(1, n_epochs + 1):
            with tqdm(train_loader, leave=False, file=sys.stdout, dynamic_ncols=True, ascii=True) as pbar:

                train_loss_r, train_loss_f, train_loss_g = 0, 0, 0
                for i, (real_images, _) in enumerate(pbar):
                    batch_size = len(real_images)
                    real_labels = torch.ones((batch_size, 1)).to(self.device)
                    fake_labels = torch.zeros((batch_size, 1)).to(self.device)
                    noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
                    real_images = real_images.to(self.device)
                    fake_images = self.generator(noise)

                    ## Training Discriminator
                    pred_r = self.discriminator(real_images)
                    loss_r = self.loss_fn(pred_r, real_labels)
                    loss_r.backward()

                    pred_f = self.discriminator(fake_images.detach())
                    loss_f = self.loss_fn(pred_f, fake_labels)
                    loss_f.backward()

                    self.optimizerD.step()
                    self.optimizerD.zero_grad()

                    # Training Generator
                    pred_g = self.discriminator(fake_images)
                    loss_g = self.loss_fn(pred_g, real_labels)
                    loss_g.backward()

                    self.optimizerG.step()
                    self.optimizerG.zero_grad()
                    
                    train_loss_r += loss_r.item()
                    train_loss_f += loss_f.item()
                    train_loss_g += loss_g.item()

                    desc = f"[{epoch:3d}/{n_epochs}] loss_r: {train_loss_r/(i + 1):.2e} " \
                           f"loss_f: {train_loss_f/(i + 1):.2e} loss_g: {train_loss_g/(i + 1):.2e}"

                    if i % 10 == 0:
                        pbar.set_description(desc)

                if epoch % step_size == 0:
                    print(desc)
                    if output_dir is not None:
                        output_images = self.generator(fixed_noise)
                        save_image(output_images, os.path.join(output_dir, f"fake_image_{epoch}.png"), normalize=True)
```

```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs = 10
learning_rate = 1E-4
latent_dim = 32

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
loss_fn = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
gan = GanTrainer(generator, discriminator, optimizerG, optimizerD)

output_dir = 'd:\\Non_Documents\\lectures\\office\\cifar10\\output_gan'
gan.fit(train_loader, n_epochs, output_dir=output_dir, step_size=2)
```

```python
from torchvision.utils import save_image
from tqdm import tqdm

torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_epochs = 10
learning_rate = 0.0002
latent_dim = 100
step_size = 2
n_outputs = 64

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
loss_fn = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

fixed_noise = torch.randn(n_outputs, latent_dim, 1, 1).to(device)
output_dir = 'd:\\Non_Documents\\lectures\\office\\cifar10\\output_gan'
output_images = generator(fixed_noise)
save_image(output_images, os.path.join(output_dir, "fake_image_0.png"), normalize=True)

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
            save_image(output_images, os.path.join(output_dir, f"fake_image_{epoch}.png"), normalize=True)
```
