import numpy as np

dim = int(3)

def wrap(phi):
    return np.round(phi / (2 * np.pi)).astype(int)

def grad(psi, a: int):
    return np.diff(psi, axis=a)

def wrap_grad(psi, a: int):
    return wrap(grad(psi, a))

def residuals(psi, a: int):
    assert(a >= 0 and a < dim)
    ax, ay = (1 + np.arange(dim-1) + a) % dim
    gx = wrap_grad(psi, a=ax)
    gy = wrap_grad(psi, a=ay)
    return grad(gy, a=ax) - grad(gx, a=ay)

def residual_loops(psi):
    rx = residuals(psi, 0)
    ry = residuals(psi, 1)
    rz = residuals(psi, 2)