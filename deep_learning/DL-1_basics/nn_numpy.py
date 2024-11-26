import numpy as np
from scipy.special import expit


def one_hot(y, n_classes):
    return np.eye(n_classes)[y]


## Activation functions
def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    return expit(x)


def softmax(x) :
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)

## Loss functions
def mse_loss(y_pred, y):
    return np.mean((y_pred - y)**2)


def bce_loss(y_pred, y):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return -np.mean(y * np.log(y_pred + 1e-7))


def cross_entropy_loss(y_pred, y):
    if y.ndim == 2:     # y: one-hot encoding (N, n_classes)
        return bce_loss(y_pred, y)  
    else:               # y: label (N, 1)
        batch_size = y_pred.shape[0]
        n_classes = y_pred.shape[1]
        return -np.mean(np.log(y_pred[np.arange(batch_size), y] + 1e-7)) / n_classes


## Metrics
def rsme(y_pred, y):
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)


def binary_accuracy(y_pred, y):
    y_pred = (y_pred > 0.5).astype(int)
    return np.mean(y_pred == y)


def accuracy(y_pred, y):
    if y.ndim == 2:
        y = y.argmax(1)
    return np.mean(y_pred.argmax(1) == y)


## Neural layers
class Module:
    def __init__(self):
        self.params = []
        self.grads = []
        self.x = None
        self.out = None

    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.w = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.b = np.zeros(output_size)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        self.params += [self.w, self.b]
        self.grads += [self.grad_w, self.grad_b]

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dout):
        self.grad_w[...] = np.dot(self.x.T, dout)   # deepcopy
        self.grad_b[...] = np.sum(dout, axis=0)     # deepcopy
        return np.dot(dout, self.w.T)               # dx


## Activation layers
class Sigmoid(Module):
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * (1 - self.out) * self.out


class ReLU(Module):
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Tanh(Module):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1 - self.out) * (1 + self.out)


## Loss functions
class MSELoss:
    def __call__(self, y_pred, y):
        loss = mse_loss(y_pred, y)
        dout = 2 * (y_pred - y) / len(y)
        return loss, dout


class BCELoss:
    def __call__(self, y_pred, y):      # y_pred: (N, 1), y: (N, 1)
        loss = bce_loss(y_pred, y)
        dout = (y_pred - y) / y_pred / (1 - y_pred) / len(y)
        return loss, dout


class CELossWithLogit:
    def __call__(self, y_pred, y):      # y_pred: (N, n_classes), # y: (N, n_classes)
        loss = cross_entropy_loss(softmax(y_pred), y)
        dout = (y_pred - y) / len(y)
        return loss, dout


## Optimizers
class Optimizer:
    def __init__(self, model, lr):
        self.params = model.params
        self.grads = model.grads
        self.lr = lr

    def step(self):
        pass


class SGD(Optimizer):
    def step(self):
        for param, grad in zip(self.params, self.grads):
            param -= self.lr * grad


class AdaGrad(Optimizer):
    def __init__(self, model, lr):
        super().__init__(model, lr)
        self.hs = []
        for param in self.params:
            self.hs.append(np.zeros_like(param))

    def step(self):
        for param, grad, h in zip(self.params, self.grads, self.hs):
            h += grad**2
            param -= self.lr * grad / (np.sqrt(h) + 1e-7)


class RMSProp(Optimizer):
    def __init__(self, model, lr, rho=0.99):
        super().__init__(model, lr)
        self.rho = rho
        self.hs = []
        for param in self.params:
            self.hs.append(np.zeros_like(param))

    def step(self):
        for param, grad, h in zip(self.params, self.grads, self.hs):
            h *= self.rho
            h += (1 - self.rho) * grad**2
            param -= self.lr * grad / (np.sqrt(h) + 1e-7)


class Adam(Optimizer):
    def __init__(self, model, lr, beta1=0.9, beta2=0.999):
        super().__init__(model, lr)
        self.beta1, self.beta2 = beta1, beta2
        self.ms, self.vs = [], []
        for param in self.params:
            self.ms.append(np.zeros_like(param))
            self.vs.append(np.zeros_like(param))
        self.t = 0

    def step(self):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for param, grad, m, v in zip(self.params, self.grads, self.ms, self.vs):
            m += (1 - self.beta1) * (grad - m)
            v += (1 - self.beta2) * (grad**2 - v)
            param -= lr_t * m / (np.sqrt(v) + 1e-7)


## Multi-layer Neurals Networks
class MLP(Module):
    def __init__(self, layer_sizes, activation, final_activation=None):
        super().__init__()
        self.layers = []
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(activation)

        self.layers.append(Linear(layer_sizes[-2], layer_sizes[-1]))
        if final_activation is not None:
            self.layers.append(final_activation)

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dout=1):
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)
        return dout


if __name__ == "__main__":

    class TwoLayerNet(Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.layers = [
                Linear(input_size, hidden_size),
                Sigmoid(),
                Linear(hidden_size, output_size),
            ]

            for layer in self.layers:
                self.params += layer.params
                self.grads += layer.grads

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def backward(self, dout=1):
            for layer in self.layers[::-1]:
                dout = layer.backward(dout)
            return dout

    ## Data
    x = np.array([[1, 2], [2, 3], [3, 4]]).astype(np.float32)
    y = np.array([[4], [7], [10]]).astype(np.float32)

    ## Model
    input_size, hidden_size, output_size = 2, 100, 1
    model = TwoLayerNet(input_size, hidden_size, output_size)
    # model = MLP(layer_sizes=[input_size, hidden_size, output_size],
    #             activation=Sigmoid())

    ## Train
    n_epochs = 1000
    learning_rate = 0.01

    # loss_fn = MSELoss()
    # metric_fn = rsme
    optimizer = SGD(model, lr=learning_rate)
    # optimizer = AdaGrad(model, lr=learning_rate)
    # optimizer = RMSProp(model, lr=learning_rate)
    # optimizer = Adam(model, lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        # Forward propagation
        y_pred = model(x)
        loss, dout = MSELoss()(y_pred, y)
        score = rsme(y_pred, y)

        # Backward progapation
        model.backward(dout)

        # Update weights and biases
        optimizer.step()

        if epoch % (n_epochs // 10) == 0:
            print(f"[{epoch:4d}/{n_epochs}] loss: {loss.item():.3f} score: {score:.3f}")


    ## ==============================================================
    ## Manual training
    ## ==============================================================
    print("\n Manual Training:")

    ## Data
    x = np.array([[1, 2], [2, 3], [3, 4]]).astype(np.float32)
    y = np.array([[4], [7], [10]]).astype(np.float32)

    ## Model
    input_size, hidden_size, output_size = 2, 100, 1

    w1 = np.ones((input_size, hidden_size)) * 0.01
    b1 = np.zeros(hidden_size)
    w2 = np.ones((hidden_size, output_size)) * 0.01
    b2 = np.zeros(output_size)

    ## Train
    n_epochs = 1000
    learning_rate = 0.01

    for epoch in range(1, n_epochs + 1):
        # Forward propagation
        z1 = np.dot(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        out = z2

        loss = np.mean((out - y)**2)
        score = rsme(out, y)

        # Backward progapation
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
            print(f"[{epoch:4d}/{n_epochs}] loss: {loss.item():.3f} score: {score:.3f}")
