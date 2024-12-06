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

data_dir = r"D:\Non_Documents\datasets\mnist"

x_train_np = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
y_train_np = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
x_test_np = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
y_test_np = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")

print(f">> Train images: {x_train_np.shape}, {x_train_np.dtype}")
print(f">> Train labels: {y_train_np.shape}, {y_train_np.dtype}")
print(f">> Test images:  {x_test_np.shape}, {x_test_np.dtype}")
print(f">> Test labels:  {y_test_np.shape}, {y_test_np.dtype}")
```

```python
## Preprocessing
x_train = x_train_np.astype(np.float32).reshape(-1, 28*28) / 255
y_train = y_train_np.astype(np.int64)

x_test = x_test_np.astype(np.float32).reshape(-1, 28*28) / 255
y_test = y_test_np.astype(np.int64)
```

```python
def set_seed(seed):
    np.random.seed(seed)

def one_hot(y, n_classes):
    return np.eye(n_classes)[y]

def accuracy(y_pred, y):
    y_pred = softmax(y_pred).argmax(axis=-1)
    return (y_pred == y).astype(np.float32).mean()

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def mean_squared_error(y_pred, y):
    return np.mean((y_pred - y)**2)

def binary_cross_entropy(y_pred, y):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return -np.mean(y * np.log(y_pred + 1e-7))

def cross_entropy(y_pred, y):
    if y.ndim == 2:     # y: one-hot encoding (N, n_classes)
        return binary_cross_entropy(y_pred, y)
    else:               # y: label (N, 1)
        batch_size = y_pred.shape[0]
        n_classes = y_pred.shape[1]
        return -np.mean(np.log(y_pred[np.arange(batch_size), y] + 1e-7)) / n_classes

from scipy.special import expit as sigmoid

class Module:
    def __init__(self):
        self.params = []
        self.grads = []

    def __call__(self, *args):
        return self.forward(*args)

class Linear(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.w = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.b = np.zeros(output_size)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        self.params = [self.w, self.b]
        self.grads = [self.grad_w, self.grad_b]

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.w) + self.b

    def backward(self, dout):
        self.grad_w[...] = np.matmul(self.x.T, dout)
        self.grad_b[...] = np.sum(dout, axis=0)
        return np.matmul(dout, self.w.T)

class Sigmoid(Module):
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)

class MLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = [
            Linear(input_size, hidden_size),
            Sigmoid(),
            Linear(hidden_size, output_size)
        ]
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dout):
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)
        return dout

class MeanSquaredError:
    def __call__(self, y_pred, y, grad=False):
        loss = mean_squared_error(y_pred, y)
        dout = 2 * (y_pred - y) / len(y)
        return (loss, dout) if grad else loss

class BinaryCrossEntropy:
    def __call__(self, y_pred, y, grad=False):
        loss = binary_cross_entropy(y_pred)
        dout = (y_pred - y) / y_pred / (1 - y_pred) / len(y)
        return (loss, dout) if grad else loss

class CrossEntoryWithLogit:
    def __call__(self, logit, y, grad=False):
        y_onehot = one_hot(y, n_classes=logit.shape[-1])
        loss = cross_entropy(softmax(logit), y_onehot)
        dout = (logit - y_onehot) / len(y)
        return (loss, dout) if grad else loss
```

```python
## Hyperparameters
batch_size = 64
learning_rate = 1e-3
n_epochs = 10

set_seed(42)
model = MLP(784, 100, 10)
loss_fn = CrossEntoryWithLogit()

for epoch in range(1, n_epochs + 1):
    ## Training
    train_loss = train_acc = 0
    indices = np.random.permutation(len(x_train))
    _x_train, _y_train = x_train[indices], y_train[indices]

    for idx in range(len(x_train) // batch_size):
        x = _x_train[idx*batch_size:(idx + 1)*batch_size]
        y = _y_train[idx*batch_size:(idx + 1)*batch_size]

        logit = model(x)
        loss, dout = loss_fn(logit, y, grad=True)
        acc = accuracy(softmax(logit), y)
        
        model.backward(dout)
        
        for param, grad in zip(model.params, model.grads):
            param -= learning_rate * grad
            
        train_loss += loss
        train_acc += acc

    ## Validation        
    test_loss = test_acc = 0
    indices = np.random.permutation(len(x_test))
    _x_test, _y_test = x_test[indices], y_test[indices]

    for idx in range(len(x_test) // batch_size):
        x = _x_test[idx*batch_size:(idx + 1)*batch_size]
        y = _y_test[idx*batch_size:(idx + 1)*batch_size]

        logit = model(x)
        test_loss += loss_fn(logit, y)
        test_acc += accuracy(softmax(logit), y)
        
    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:3d}/{n_epochs}] "
              f"loss: {train_loss/(len(x_train) // batch_size):.3f} acc: {train_acc/(len(x_train) // batch_size):.3f} | "
              f"val_loss: {test_loss/(len(x_test) // batch_size):.3f} acc: {test_acc/(len(x_test) // batch_size):.3f}")
```

```python
class DataLoader:
    def __init__(self, data, labels, batch_size, shuffle=False):
        indices = np.random.permutation(len(data))
        self.data = data[indices] if shuffle else data
        self.labels = labels[indices] if shuffle else labels

        self.batch_size = batch_size
        # self.n_batches = len(data) // batch_size
        self.n_batches = int(np.ceil(len(data) / batch_size))

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for i in range(self.n_batches):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            yield self.data[start:end], self.labels[start:end]

batch_size = 64
train_loader = DataLoader(x_train, y_train, batch_size, shuffle=True)
test_loader = DataLoader(x_test, y_test, batch_size, shuffle=False)

x, y = next(iter(train_loader))
x.shape, y.shape

class Optimizer:
    def __init__(self, model, lr):
        self.params = model.params
        self.grads = model.grads
        self.lr = lr

class SGD(Optimizer):
    def step(self):
        for param, grad in zip(self.params, self.grads):
            param -= self.lr * grad

class Adam(Optimizer):
    def __init__(self, model, lr, beta1=0.0, beta2=0.999):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0

        self.ms, self.vs = [], []
        for param in self.params:
            self.ms.append(np.zeros_like(param))
            self.vs.append(np.zeros_like(param))

    def step(self):
        self.iter += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.iter) / (1 - self.beta1**self.iter)
        for param, grad, m, v in zip(self.params, self.grads, self.ms, self.vs):
            m += (1 - self.beta1) * (grad - m)
            v += (1 - self.beta2) * (grad**2 - v)
            param -= lr_t * m / (np.sqrt(v) + 1e-7)
```

```python
## Hyperparameters
learning_rate = 1e-3
n_epochs = 10

set_seed(111)
model = MLP(784, 100, 10)
optimizer = Adam(model, lr=learning_rate)
loss_fn = CrossEntoryWithLogit()

for epoch in range(1, n_epochs + 1):
    ## Training
    train_loss = train_acc = 0
    for x, y in train_loader:
        logit = model(x)
        loss, dout = loss_fn(logit, y, grad=True)
        acc = accuracy(softmax(logit), y)

        model.backward(dout)
        optimizer.step()

        train_loss += loss
        train_acc += acc

    ## Validation
    test_loss = test_acc = 0
    for x, y in test_loader:
        logit = model(x)
        test_loss += loss_fn(logit, y)
        test_acc += accuracy(softmax(logit), y)

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:3d}/{n_epochs}] "
              f"loss: {train_loss/len(train_loader):.3f} acc: {train_acc/len(train_loader):.3f} | "
              f"val_loss: {test_loss/len(test_loader):.3f} val_acc: {test_acc/len(test_loader):.3f}")
```

```python
class Trainer:
    def __init__(self, model, optimizer, loss_fn, metric_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn

    def train_step(self, x, y):
        y_pred = self.model(x)
        loss, dout = self.loss_fn(y_pred, y, grad=True)
        acc = self.metric_fn(y_pred, y)

        self.model.backward(dout)
        self.optimizer.step()
        return {"loss": loss, "acc": acc}

    def test_step(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        acc = self.metric_fn(y_pred, y)
        return {"loss": loss, "acc": acc}

    def fit(self, train_loader, n_epochs, valid_loader):
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        for epoch in range(1, n_epochs + 1):
            ## Training
            train_loss = train_acc = 0
            for x, y in train_loader:
                results = self.train_step(x, y)
                train_loss += results["loss"]
                train_acc += results["acc"]

            history["loss"].append(train_loss / len(train_loader))
            history["acc"].append(train_acc / len(train_loader))

            ## Validation
            test_loss = test_acc = 0
            for x, y in valid_loader:
                results = self.test_step(x, y)
                test_loss += results["loss"]
                test_acc += results["acc"]
                
            history["val_loss"].append(test_loss / len(test_loader))
            history["val_acc"].append(test_acc / len(test_loader))
            
            ## Log results
            if epoch % (n_epochs // 10) == 0:
                print(f"[{epoch:3d}/{n_epochs}] "
                      f"loss: {history['loss'][-1]:.3f} acc: {history['acc'][-1]:.3f} | "
                      f"val_loss: {history['val_loss'][-1]:.3f} val_acc: {history['val_acc'][-1]:.3f}")
        return history
```

```python
import matplotlib.pyplot as plt

## Hyperparameters
learning_rate = 1e-4
n_epochs = 10

set_seed(111)
model = MLP(784, 100, 10)
optimizer = Adam(model, lr=learning_rate)
loss_fn = CrossEntoryWithLogit()
clf = Trainer(model, optimizer, loss_fn, metric_fn=accuracy)

history = clf.fit(train_loader, n_epochs, valid_loader=test_loader)
```

```python
class TrainerV2:
    def __init__(self, model, optimizer, loss_fn, metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = {"loss": loss_fn}
        if metrics is not None:
            self.metrics.update(metrics)

    def train_step(self, x, y):
        y_pred = self.model(x)
        _, dout = self.loss_fn(y_pred, y, grad=True)
        self.model.backward(dout)
        self.optimizer.step()
        return {name: func(y_pred, y) for name, func in self.metrics.items()}

    def test_step(self, x, y):
        y_pred = self.model(x)
        return {name: func(y_pred, y) for name, func in self.metrics.items()}

    def fit(self, train_loader, n_epochs, valid_loader=None):
        history = {name: [] for name in self.metrics}
        if valid_loader is not None:
            history.update({f"val_{name}": [] for name in self.metrics})

        for epoch in range(1, n_epochs + 1):
            cur_epoch = f"[{epoch:3d}/{n_epochs}]"
            ## Training
            res = {name: 0 for name in self.metrics}
            for i, (x, y) in enumerate(train_loader):
                res_step = self.train_step(x, y)

                desc = ""
                for name in self.metrics:
                    res[name] += res_step[name]
                    desc += f" {name}: {res[name] / (i + 1):.3f}"

            for name in self.metrics:
                history[name].append(res[name] / len(train_loader))
            
            if valid_loader is None:    
                if epoch % (n_epochs // 10) == 0:
                    print(cur_epoch + desc)
                continue

            ## Validation
            res = {name: 0 for name in self.metrics}
            for i, (x, y) in enumerate(valid_loader):
                res_step = self.test_step(x, y)

                val_desc = ""
                for name in self.metrics:
                    res[name] += res_step[name]
                    val_desc += f" val_{name}: {res[name] / (i + 1):.3f}"

            for name in self.metrics:
                history[f"val_{name}"].append(res[name] / len(valid_loader))

            if epoch % (n_epochs // 10) == 0:
                print(cur_epoch + desc + " |" + val_desc)
                
        return history
```

```python
## Hyperparameters
learning_rate = 1e-3
n_epochs = 10

set_seed(111)
model = MLP(784, 100, 10)
optimizer = SGD(model, lr=learning_rate)
loss_fn = CrossEntoryWithLogit()
metrics = {"acc": accuracy}
clf = TrainerV2(model, optimizer, loss_fn, metrics=metrics)

history = clf.fit(train_loader, n_epochs, valid_loader=test_loader)
```
