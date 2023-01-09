"""
SYNOPSIS
    Helper functions for data generation and value function network training
DESCRIPTION

    Contains helper functions such as computing the input for the ILQR system,
    tracking costs and offseting angles to be between 0 and 2pi.
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.0
"""

from itertools import accumulate
import numpy as np

gamma = 0.99


def compute_input(x, r, rdot, Kp, Kd):
    """
    Function to compute the PD control law
    """
    theta = x[2]
    phi = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    v, w = np.matmul(np.linalg.inv(np.eye(2) - np.matmul(Kd, phi)), np.matmul(Kp, x - r) - np.matmul(Kd, rdot))
    return np.array([v, w])


def angle_wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def compute_cum_tracking_cost(ref_traj, actual_traj, Kp, Kd, N):
    input_traj = [compute_input(x, r, rdot, Kp, Kd) for x, r, rdot in
                  zip(actual_traj, ref_traj[:, 0:3], ref_traj[:, 3:])]
    xcost = [np.linalg.norm(actual_traj[i:i + N, :2] - ref_traj[i:i + N, :2], axis=1) ** 2 +
             angle_wrap(actual_traj[i:i + N, 2] - ref_traj[i:i + N, 2]) ** 2 for i in range(len(ref_traj) - N)]

    xcost.reverse()
    cost = []
    for i in range(len(ref_traj) - N):
        tot = list(accumulate(xcost[i], lambda x, y: x * gamma + y))
        cost.append(tot[-1])
    cost.reverse()
    return np.vstack(cost), np.vstack(input_traj)






