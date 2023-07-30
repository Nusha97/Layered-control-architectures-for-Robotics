"""
SYNOPSIS
    Helper functions for data generation and value function network training

DESCRIPTION
    Contains helper functions such as computing the input for the ILQR system,
    tracking costs and offseting angles to be between 0 and 2pi.

AUTHOR
    Anusha Srikanthan <sanusha@seas.upenn.edu>

VERSION
    0.0
"""

from itertools import accumulate
import numpy as np

gamma = 1
Kp = 5 * np.array([[2, 1, 0], [0, 1, 3]])

def compute_input(x, r, rdot):
    """
    Function to compute the PD control law
    :param x: state of the unicycle model
    :param r: reference to track
    :param rdot: input of the closed loop system
    :return: v, w
    """
    theta = angle_wrap(x[2])
    if theta == 0:
        theta = 0.01
    phi = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    Kd = np.linalg.pinv(phi)
    v, w = np.matmul(Kd, rdot) + np.matmul(Kp, x-r)
    return np.array([v, w])


def compute_rdot(ref, dt):
    """
    Return the numerical differentiation of ref
    :param ref: reference to track
    :param dt: discretization step size
    :return: rdot
    """
    cur_ref = ref[1:]
    prev_ref = ref[:-1]
    rdot = (cur_ref - prev_ref) / dt
    return rdot


def angle_wrap(theta):
    """
    Function to wrap angles greater than pi
    :param theta: heading angle of unicycle
    :return: wrapped angle
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def compute_tracking_cost(ref_traj, actual_traj, rdot_traj, horizon, rho=0):
    """
    Tracking cost function as defined in our paper (x - r) ** 2 + u ** 2
    :param ref_traj: Reference trajectories from the dataset
    :param actual_traj: System rollouts from forward simulation
    :param rdot_traj: Input of the closed loop unicycle system
    :param horizon: horizon length of the trajectory
    :param rho: tracking penalty factor
    :return: cost, input
    """
    num_traj = int(ref_traj.shape[0]/horizon)
    input_traj = []

    for i in range(len(ref_traj)-num_traj):
        input_traj.append(compute_input(actual_traj[i, :], ref_traj[i, :], rdot_traj[i-1, :]))
    input_traj.append(np.zeros(3))
    xcost = []
    for i in range(num_traj):
        act = actual_traj[i*horizon:(i+1)*horizon, :]
        act = np.append(act, act[-1, :] * np.ones((horizon-1, 3)))
        act = np.reshape(act, (horizon*2-1, 3))
        r0 = ref_traj[i*horizon:(i+1)*horizon, :]
        r0 = np.append(r0, r0[-1, :] * np.ones((horizon-1, 3)))
        r0 = np.reshape(r0, (horizon * 2 - 1, 3))

        xcost.append(rho * (np.linalg.norm(act[:, :2] - r0[:, :2], axis=1) ** 2 +
             rho * angle_wrap(act[:, 2] - r0[:, 2]) ** 2) + 0.00001 * np.linalg.norm(input_traj[i]) ** 2)


    xcost.reverse()
    cost = []
    for i in range(num_traj):
        tot = list(accumulate(xcost[i], lambda x, y: x * gamma + y))
        if tot[-1] > 0:
            cost.append(np.log(tot[-1]))
        else:
            cost.append(np.log(1e-10))
    cost.reverse()
    return np.vstack(cost), rdot_traj
