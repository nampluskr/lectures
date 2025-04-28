```python
import numpy as np
import matplotlib.pyplot as plt
```

### $y = ax + b$

```python
n_samples= 101
a, b = 2.5, -5

x = np.linspace(0, 10, n_samples)
y = a * x + b

x_noise = 0.2 * np.random.randn(n_samples)
y_noise = 0.5 * np.random.randn(n_samples)

x_data = x + x_noise
y_data = a * x_data + b + y_noise

plt.plot(x_data, y_data, 'x')
plt.plot(x, y)
plt.show()
```

```python
# Analytical Solution
x, y = x_data, y_data
X = np.vstack([x, np.ones_like(x)]).T

# Normal Equation: (X^T X)^(-1) X^T y
params = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
params
```

```python
learning_rate = 0.01
n_epoches = 1000

def model(x, a, b):
    return a * x + b

a, b = 0.1, 0.1
x, y = x_data, y_data

for epoch in range(1, n_epoches + 1):
    y_pred = model(x, a, b)
    loss = np.mean((y - y_pred)**2)
    
    da = -2 * np.mean(x * (y - y_pred))
    db = -2 * np.mean(y - y_pred)
    
    a -= learning_rate * da
    b -= learning_rate * db
    
    if epoch % 100 == 0:
        print(f"Epoch[{epoch:5d}/{n_epoches}] loss={loss:.3f}, a={a:.3f}, b={b:.3f}")
```

## $y = ax^2 + bx + c$

```python
n_samples= 101
a, b, c = 1.5, -2.0, 3.0

x = np.linspace(-5, 5, n_samples)
y = a * x**2 + b * x + c

x_noise = 0.1 * np.random.randn(n_samples)
y_noise = 0.5 * np.random.randn(n_samples)

x_data = x + x_noise
y_data = a * x_data**2 + b * x_data + c + y_noise

plt.plot(x_data, y_data, 'x')
plt.plot(x, y)
plt.show()
```

```python
# Analytical Solution
x, y = x_data, y_data
X = np.vstack([x**2, x, np.ones_like(x)]).T

# Normal Equation: (X^T X)^(-1) X^T y
params = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
params
```

```python
learning_rate = 0.001
n_epoches = 5000

def model(x, a, b, c):
    return a * x**2 + b * x + c

a, b, c = 0.1, 0.1, 0.1
x, y = x_data, y_data

for epoch in range(1, n_epoches + 1):
    y_pred = model(x, a, b, c)
    loss = np.mean((y - y_pred)**2)
    
    da = -2 * np.mean(x**2 * (y - y_pred))
    db = -2 * np.mean(x * (y - y_pred))
    dc = -2 * np.mean(y - y_pred)
    
    a -= learning_rate * da
    b -= learning_rate * db
    c -= learning_rate * dc
    
    if epoch % 500 == 0:
        print(f"Epoch[{epoch:5d}/{n_epoches}] loss={loss:.3f}, a={a:.3f}, b={b:.3f}, c={c:.3f}")
```
