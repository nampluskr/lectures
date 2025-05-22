```python
import numpy as np
import matplotlib.pyplot as plt

# 결정계수: 추정한 회귀 모델이 주어진 자료에 적합한 정도를 재는 척도
def r2_score(y_true, y_pred):
    sst = np.sum((y_true - y_true.mean())**2)    # Total sum of squares
    ssr = np.sum((y_true - y_pred)**2)           # Residual sum of squares
    return 1 - (ssr / sst)
```

### $y = ax$

```python
def model(x, a):
    return a * x

n_samples = 101
a = 2.5
x = np.linspace(0, 5, n_samples)
y = model(x, a)

np.random.seed(42)
x_noise = 0.2 * np.random.randn(n_samples)
y_noise = 0.5 * np.random.randn(n_samples)

x_data = x + x_noise
y_data = model(x_data, a) + y_noise

plt.plot(x_data, y_data, 'ko', ms=5)
plt.plot(x, y, 'r--', lw=2)
plt.show()
```

```python
# Analytical Solution
x, y = x_data, y_data
X = np.vstack([x]).T

# Normal Equation: (X^T X)^(-1) X^T y
params = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
params
```

```python
learning_rate = 0.001
n_epoches = 1000

a = 0.1                 # Model parameters (initial value)
x, y = x_data, y_data   # data

for epoch in range(1, n_epoches + 1):
    y_pred = model(x, a)
    loss = np.mean((y - y_pred)**2)
    score = r2_score(y, y_pred)

    grad_a = -2 * np.mean(x * (y - y_pred))

    a -= learning_rate * grad_a

    if epoch % 100 == 0:
        print(f"Epoch[{epoch:4d}/{n_epoches}] loss={loss:.4f}, score={score:.4f}, a={a:.4f}")
```


### $y = ax + b$

```python
def model(x, a, b):
    return a * x + b

n_samples = 101
a, b = 2.5, -5
x = np.linspace(-5, 5, n_samples)
y = model(x, a, b)

np.random.seed(42)
x_noise = 0.2 * np.random.randn(n_samples)
y_noise = 0.5 * np.random.randn(n_samples)

x_data = x + x_noise
y_data = model(x_data, a, b) + y_noise

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

a, b = 0.1, 0.1         # Model parameters
x, y = x_data, y_data   # data

for epoch in range(1, n_epoches + 1):
    y_pred = model(x, a, b)
    loss = np.mean((y - y_pred)**2)
    score = r2_score(y, y_pred)
    
    grad_a = -2 * np.mean(x * (y - y_pred))
    grad_b = -2 * np.mean(y - y_pred)
    
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    
    if epoch % 100 == 0:
        print(f"Epoch[{epoch:4d}/{n_epoches}] loss={loss:.3f}, score={score:.3f}, a={a:.3f}, b={b:.3f}")
```

## $y = ax^2 + bx + c$

```python
def model(x, a, b, c):
    return a * x**2 + b * x + c

n_samples= 101
a, b, c = 1.5, -2.0, 3.0

x = np.linspace(-5, 5, n_samples)
y = model(x, a, b, c)

np.random.seed(42)
x_noise = 0.1 * np.random.randn(n_samples)
y_noise = 0.5 * np.random.randn(n_samples)

x_data = x + x_noise
y_data = model(x_data, a, b, c) + y_noise

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

a, b, c = 0.1, 0.1, 0.1
x, y = x_data, y_data

for epoch in range(1, n_epoches + 1):
    y_pred = model(x, a, b, c)
    loss = np.mean((y - y_pred)**2)
    score = r2_score(y, y_pred)
    
    grad_a = -2 * np.mean(x**2 * (y - y_pred))
    grad_b = -2 * np.mean(x * (y - y_pred))
    grad_c = -2 * np.mean(y - y_pred)
    
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    
    if epoch % 500 == 0:
        print(f"Epoch[{epoch:5d}/{n_epoches}] loss={loss:.3f}, score={score:.3f}, a={a:.3f}, b={b:.3f}, c={c:.3f}")
```
