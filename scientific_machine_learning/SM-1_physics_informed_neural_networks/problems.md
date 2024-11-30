## Problems for Physics-informed Neural Networks

### Harmonic Damping Oscillator

$$m\frac{d^2u}{dt^2} + \mu\frac{du}{dt} + k,\quad t\in[0, 1]$$

- Initial conditions:

$$u(0)=1,\quad \frac{du}{dt}(0) = 0$$

### Burgers' Equation

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial t} - \nu\frac{\partial^2 u}{\partial t^2} = 0, \quad\nu=\frac{0.01}{\pi}, \quad x\in [-1, 1],\quad t\in [0, 1]$$

- Initial conditon:

$$u(x, 0) = -\sin(\pi x)$$

- Boundary conditions:

$$u(-1, t) = 0,\quad u(1, t) = 0$$