```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import nn_numpy as nn
```

## Regression: Diabetes

```python
## Regression - Diabetes
from sklearn.datasets import load_diabetes

x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(">> Train:", x_train_scaled.shape, y_train.shape)
print(">> Test: ", x_test_scaled.shape, y_test.shape)

x, y = x_train_scaled, y_train.reshape(-1, 1)
print(x.shape, y.shape)
```

```python
## Model
np.random.seed(42)
input_size, hidden_size, output_size = 10, 100, 1

w1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
b1 = np.zeros(hidden_size)
w2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
b2 = np.zeros(output_size)

## Train
n_epochs = 1000
learning_rate = 0.01

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    z1 = np.dot(x, w1) + b1
    a1 = nn.sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    out = z2

    loss = np.mean((out - y)**2)
    score = nn.rsme(out, y)

    # Backward propagation
    grad_out = 2 * (out - y) / y.shape[0]
    grad_z2 = grad_out
    grad_w2 = np.dot(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0)

    grad_a1 = np.dot(grad_z2, w2.T)
    grad_z1 = a1 * (1 - a1) * grad_a1
    grad_w1 = np.dot(x.T, grad_z1)
    grad_b1 = np.sum(grad_z1, axis=0)

    # Update weights and biases
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] loss: {loss.item():.2f} score: {score:.4f}")
```

```python
# Model
np.random.seed(42)
model = nn.MLP(layer_sizes=[10, 100, 1], activation=nn.Sigmoid())

## Train
n_epochs = 1000
learning_rate = 0.01
optimizer = nn.SGD(model, lr=learning_rate)

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    y_pred = model(x)
    loss, dout = nn.MSELoss()(y_pred, y)
    score = nn.rsme(y_pred, y)

    # Backward propagation
    model.backward(dout)
    
    # Update weights and biases
    optimizer.step()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] loss: {loss.item():.2f} score: {score:.4f}")
```

## Binary Classification (Logistic Regression): Breast Cancer

```python
from sklearn.datasets import load_breast_cancer

# Load data
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(">> Train:", x_train_scaled.shape, y_train.shape)
print(">> Test: ", x_test_scaled.shape, y_test.shape)

x, y = x_train_scaled, y_train.reshape(-1, 1)
print(x.shape, y.shape)
```

```python
## Model
np.random.seed(42)
input_size, hidden_size, output_size = 30, 100, 1

w1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
b1 = np.zeros(hidden_size)
w2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
b2 = np.zeros(output_size)

## Train
n_epochs = 100
learning_rate = 0.01

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    z1 = np.dot(x, w1) + b1
    a1 = nn.sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    y_pred = out = nn.sigmoid(z2)

    loss = nn.bce_loss(y_pred, y)
    score = nn.binary_accuracy(y_pred, y)

    # Backward propagation
    dout = (out - y) / out / (1 - out) / len(y)
    grad_z2 = out * (1 - out) * dout
    grad_w2 = np.dot(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0)

    grad_a1 = np.dot(grad_z2, w2.T)
    grad_z1 = a1 * (1 - a1) * grad_a1
    grad_w1 = np.dot(x.T, grad_z1)
    grad_b1 = np.sum(grad_z1, axis=0)

    # Update weights and biases
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] loss: {loss.item():.2f} score: {score:.4f}")
```

```python
## Model
np.random.seed(42)
model = nn.MLP(layer_sizes=[30, 100, 1], activation=nn.Sigmoid(), 
               final_activation=nn.Sigmoid())

## Train
n_epochs = 100
learning_rate = 0.01
optimizer = nn.SGD(model, lr=learning_rate)

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    y_pred = model(x)
    loss, dout = nn.BCELoss()(y_pred, y)
    score = nn.binary_accuracy(y_pred, y)

    # Backward propagation
    model.backward(dout)

    # Update weights and biases
    optimizer.step()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] loss: {loss.item():.2f} score: {score:.4f}")
```

## Multiclass Classification: Iris

```python
from sklearn.datasets import load_iris

# Load data
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(">> Train:", x_train_scaled.shape, y_train.shape)
print(">> Test: ", x_test_scaled.shape, y_test.shape)

x, y = x_train_scaled, nn.one_hot(y_train, n_classes=3)
print(x.shape, y.shape)
```

```python
# Model
np.random.seed(42)
input_size, hidden_size, output_size = 4, 100, 3

w1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
b1 = np.zeros(hidden_size)
w2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
b2 = np.zeros(output_size)

## Train
n_epochs = 100
learning_rate = 0.01

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    z1 = np.dot(x, w1) + b1
    a1 = nn.sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    out = nn.softmax(z2)

    loss = nn.cross_entropy_loss(out, y)
    score = nn.accuracy(out, y)

    # Backward propagation
    grad_z2 = (z2 - y) / y.shape[0]
    grad_w2 = np.dot(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0)

    grad_a1 = np.dot(grad_z2, w2.T)
    grad_z1 = a1 * (1 - a1) * grad_a1
    grad_w1 = np.dot(x.T, grad_z1)
    grad_b1 = np.sum(grad_z1, axis=0)

    # Update weights and biases
    w1 -= learning_rate * grad_w1
    b1 -= learning_rate * grad_b1
    w2 -= learning_rate * grad_w2
    b2 -= learning_rate * grad_b2

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] loss: {loss.item():.2f} score: {score:.4f}")
```

```python
# Model
np.random.seed(42)
model = nn.MLP(layer_sizes=[4, 100, 3], activation=nn.Sigmoid())

## Train
n_epochs = 100
learning_rate = 0.01
optimizer = nn.SGD(model, lr=learning_rate)

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    logit = model(x)
    y_pred = nn.softmax(logit)
    loss, dout = nn.CELossWithLogit()(logit, y)
    score = nn.accuracy(y_pred, y)

    # Backward propagation
    model.backward(dout)
    
    # Update weights and biases
    optimizer.step()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] loss: {loss.item():.2f} score: {score:.4f}")
```
