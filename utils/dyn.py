import numpy as np

def dyn(t, state, u):
    x = state[0:1]
    x_dot = state[1:2]
    x_ddot = u + 2*(x_dot*np.sin(4*x_dot) + np.cos(x))
    # x_ddot = u
    return np.concatenate((x_dot, x_ddot))