##############################################################################
# Fengjun Yang, 2022
# Utility functions for testing the learned value and tracking controllers
##############################################################################

import numpy as np
import matplotlib.pyplot as plt

def relerr(A, Ahat):
    """ computes relative error of two matrices in terms of frobenius norm """
    return np.linalg.norm(A-Ahat, 'fro') / np.linalg.norm(A, 'fro')

def moving_sum(x, window_size, gamma=1):
    conv = np.convolve(x, gamma ** np.arange(window_size-1, -1, -1))
    return conv[window_size-1:-window_size+1]

def viztraj3D(trajs, labels=None, waypoints=None):
    ax = plt.axes(projection='3d')
    # Plot traj
    for i, traj in enumerate(trajs):
        ax.plot3D(traj[0], traj[1], traj[2], label=labels[i] if labels is not None else None)
    if waypoints is not None:
        # Plot start and goal
        ax.scatter3D(waypoints[0,0], waypoints[1,0], waypoints[2,0], color='red')
        ax.scatter3D(waypoints[0,-1], waypoints[1,-1], waypoints[2,-1], color='red')
        # Plot intermediate waypoints
        ax.scatter3D(waypoints[0,1:-1], waypoints[1,1:-1], waypoints[2,1:-1])
    ax.legend()

def rostraj2aug(actual, ref, u, Tref, gamma):
    ''' Converts ros trajectories to the format we use for training '''
    T = actual.shape[0]
    # Compute cost
    xcost = np.linalg.norm(actual[:,(0,1,2)] - ref[:,(0,1,2)], axis=1)**2 
    ucost = 0.1 * np.linalg.norm(u, axis=1)**2
    # Cost from yaw (needs special treatment because quotient norm)
    ar = np.abs(actual[:, 12] - ref[:, 12])
    ra = np.abs(actual[:, 12] + 2*np.pi-ref[:,12])
    yawcost = np.minimum(ar, ra) ** 2
    cost = xcost + yawcost + ucost
    # Compute the value for each trajectory
    vtraj = moving_sum(cost, Tref, gamma)
    # Do state augmentation
    #x_aug = [[actual[i,(0,1,2,-2)], ref[i:i+Tref, (0,1,2,-2)].flatten()] for i in range(T-Tref)]
    x_aug = [[actual[i,:], ref[i:i+Tref, :].flatten()] for i in range(T-Tref+1)]
    return np.block(x_aug), u[:T-Tref+1], cost[:T-Tref+1, None], vtraj[:, None]

def load_datasets(dsets, Tref, gamma):
    xtrajs, utrajs, rtrajs, vtrajs, xtrajs_ = [], [], [], [], []
    for actual, ref, u in dsets:
        x, u, r, v = rostraj2aug(actual, ref, u, Tref, gamma)
        xtrajs.append(x[:-1, :])
        xtrajs_.append(x[1:, :])
        utrajs.append(u[:-1, :])
        rtrajs.append(r[:-1, :])
        vtrajs.append(v[:-1, :])
    return np.vstack(xtrajs), np.vstack(xtrajs_), np.vstack(utrajs), \
            np.vstack(rtrajs), np.vstack(vtrajs)