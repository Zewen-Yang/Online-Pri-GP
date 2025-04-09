import numpy as np

def agentDynamics(t, state, u, f, m1, l1, m2, l2, g):
    x = state[0:2]
    x_dot = state[2:4]
    # Calculate system parameters
    alpha = m1 * (l1**2) + m2 * (l1**2 + (l2/2)**2)
    beta = m2 * l1 * l2/2
    delta = m2 * (l2/2)**2
    
    
    # Mass matrix inverse M
    M_mat = np.array([
        [alpha+2*beta*np.cos(x[1]), delta+beta*np.cos(x[1])],
        [delta+beta*np.cos(x[1]), alpha]
    ])
    M_inv = np.linalg.inv(M_mat)

    C_mat = np.array([
        [-beta*np.sin(x[1])*x_dot[1], -beta*np.sin(x[1])*(x_dot[0]+x_dot[1])],
        [beta*np.sin(x[1])*x_dot[0], 0]
    ])

    G_vec = np.array([
        [(m2+m1)*g*l1/2*np.sin(x[0]) + m2*g*l2/2*np.sin(x[0]+x[1])], 
        [m2*g*l2/2*np.sin(x[0]+x[1])]
    ])
    
    x_ddot = M_inv @ (u.T - C_mat @ x_dot.reshape(-1,1) - G_vec - f.T)
    return np.concatenate((x_dot, x_ddot.squeeze()))