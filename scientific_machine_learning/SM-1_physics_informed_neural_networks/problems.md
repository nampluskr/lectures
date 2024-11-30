# Problems for Physics-informed Neural Networks

## Harmonic Damping Oscillator

$$m\frac{d^2u}{dt^2} + \mu\frac{du}{dt} + k = 0,\quad t\in[0, 1]$$

- Initial conditions:

$$u(0)=1,\quad \frac{du}{dt}(0) = 0$$

- Exact solution:

$$u(t) = 2A\cos(\phi + wt)\exp(-dt)$$
where 

$$w=\sqrt{w_0^2 - d^2},\quad\phi = \tan^{-1}\left(-\frac{d}{w}\right),\quad A = \frac{1}{2\cos\phi}$$


## Heat-1D Equation

$$\frac{\partial u}{\partial t} = \alpha\frac{\partial^2 u}{\partial x^2},\quad t\in [0, T],\quad x\in [0, L]$$

- Initial condition:

$$u(0, x) = \sin\left(\frac{n\pi x}{L}\right),\quad x\in [0, L]$$

- Boundary conditions:

$$u(t, 0) = u(t, L) = 0,\quad t\in [0, T]$$

- Exact solution:

$$u(t, x) = \sin\left(\frac{n\pi}{L}x\right)\exp\left(-\alpha\frac{n^2\pi^2}{L^2}t\right)$$



## Burgers' Equation

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial t} - \nu\frac{\partial^2 u}{\partial t^2} = 0, \quad\nu=\frac{0.01}{\pi}, \quad x\in [-1, 1],\quad t\in [0, 1]$$

- Initial conditon:

$$u(x, 0) = -\sin(\pi x)$$

- Boundary conditions:

$$u(-1, t) = 0,\quad u(1, t) = 0$$