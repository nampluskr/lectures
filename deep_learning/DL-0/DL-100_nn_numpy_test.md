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
```

```python
## Preprocess
x, y = x_train_scaled, y_train.reshape(-1, 1)
print(x.shape, y.shape)

## Model
np.random.seed(42)
input_size, hidden_size, output_size = 10, 100, 1

w1 = np.random.randn(input_size, hidden_size)   # weight of 1st layer
b1 = np.zeros(hidden_size)                      # bias of 1st layer
w2 = np.random.randn(hidden_size, output_size)  # weight of 2nd layer
b2 = np.zeros(output_size)                      # bias of 2nd layer

## Train
n_epochs = 10000
learning_rate = 0.1

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    a0 = x
    z1 = np.dot(a0, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = identity(z2)

    loss = np.mean((a2 - y)**2)
    score = r2_score(a2, y)

    # Backward propagation
    grad_a2 = 2 * (a2 - y) / len(y)
    grad_z2 = grad_a2 * 1
    grad_a1 = np.dot(grad_z2, w2.T)
    grad_z1 = grad_a1 * a1 * (1 - a1)
    grad_a0 = np.dot(grad_z1, w1.T)

    grad_w2 = np.dot(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0)
    grad_w1 = np.dot(a0.T, grad_z1)
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
## Preprocess
x, y = x_train_scaled, y_train.reshape(-1, 1)
print(x.shape, y.shape)

## Model
np.random.seed(42)
input_size, hidden_size, output_size = 30, 100, 1

w1 = np.random.randn(input_size, hidden_size)   # weight of 1st layer
b1 = np.zeros(hidden_size)                      # bias of 1st layer
w2 = np.random.randn(hidden_size, output_size)  # weight of 2nd layer
b2 = np.zeros(output_size)                      # bias of 2nd layer

## Train
n_epochs = 10000
learning_rate = 0.01

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    a0 = x
    z1 = np.matmul(a0, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.matmul(a1, w2) + b2
    a2 = sigmoid(z2)

    loss = binary_cross_entropy(a2, y)
    score = binary_accuracy(a2, y)

    # Backward propagation
    grad_a2 = (a2 - y) / a2 / (1 - a2) / len(y)
    grad_z2 = grad_a2 * a2 * (1 - a2)
    grad_a1 = np.matmul(grad_z2, w2.T)
    grad_z1 = grad_a1 * a1 * (1 - a1)
    grad_a0 = np.matmul(grad_z1, w1.T)

    grad_w2 = np.matmul(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0)
    grad_w1 = np.matmul(a0.T, grad_z1)
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
## Preprocess
x, y = x_train_scaled, one_hot(y_train, n_classes=3)
print(x.shape, y.shape)

## Model
np.random.seed(42)
input_size, hidden_size, output_size = 4, 100, 3
 
w1 = np.random.randn(input_size, hidden_size)   # weight of 1st layer
b1 = np.zeros(hidden_size)                      # bias of 1st layer
w2 = np.random.randn(hidden_size, output_size)  # weight of 2nd layer
b2 = np.zeros(output_size)                      # bias of 2nd layer

## Train
n_epochs = 1000
learning_rate = 0.01

for epoch in range(1, n_epochs + 1):
    # Forward propagation
    a0 = x
    z1 = np.matmul(a0, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.matmul(a1, w2) + b2
    a2 = softmax(z2)

    loss = cross_entropy(a2, y)
    score = accuracy(a2, y)

    # Backward propagation
    grad_z2 = (z2 - y) / len(y)
    grad_a1 = np.matmul(grad_z2, w2.T)
    grad_z1 = grad_a1 * a1 * (1 - a1)
    grad_a0 = np.matmul(grad_z1, w1.T)

    grad_w2 = np.matmul(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0)
    grad_w1 = np.matmul(a0.T, grad_z1)
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
