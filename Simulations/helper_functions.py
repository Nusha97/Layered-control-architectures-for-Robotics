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


def compute_rdot(ref, dt):
    """
    Return the numerical differentiation of ref
    :param ref:
    :param dt:
    :return:
    """
    cur_ref = ref[1:]
    prev_ref = ref[:-1]
    rdot = np.zeros(ref.shape)
    rdot[1:, :] = (cur_ref - prev_ref) / dt

    return rdot



def forward_simulate(x0, r, Kp, Kd, N):
    """
    Simulate the unicycle dynamical system for the given reference trajectory
    :param x0: initial condition
    :param r: reference trajectory
    :param N: horizon of reference
    :return:
    """
    # Compute rdot numerically
    dt = 0.01
    cur_ref = r[1:]
    prev_ref = r[:-1]

    rdot = np.zeros(r.shape)
    x = np.zeros(r.shape)
    xdot = np.zeros(r.shape)
    x[0, :] = x0

    rdot[1:, :] = (cur_ref - prev_ref)/dt
    v, w = compute_input(x0, r[0, :], rdot[0, :], Kp, Kd)
    xdot[0, :] = np.array([v * np.cos(x0[2]), v * np.sin(x0[2]), w])
    for i in range(1, N):
        x[i, :] = x[i-1, :] + xdot[i-1, :] * dt
        v, w = compute_input(x[i, :], r[i, :], rdot[i, :], Kp, Kd)
        xdot[i, :] = np.array([v * np.cos(x[i, 2]), v * np.sin(x[i, 2]), w])

    cost = np.linalg.norm(x[:, :2] - r[:, :2], axis=1) ** 2 + angle_wrap(x[:, 2] - r[:, 2]) ** 2
    # Computing the cumulative cost with gamma
    return list(accumulate(cost[::-1], lambda x, y: x * gamma + y))[-1], x
    


def angle_wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def compute_tracking_cost(ref_traj, actual_traj, rdot_traj, Kp, Kd, N):
    input_traj = []
    for i in range(len(ref_traj)):
        input_traj.append(compute_input(actual_traj[i, :], ref_traj[i, :], rdot_traj[i, :], Kp, Kd))
    # input_traj = [compute_input(x, r, rdot, Kp, Kd) for x, r, rdot in
    #              zip(actual_traj, ref_traj, rdot_traj)]
    xcost = [np.linalg.norm(actual_traj[i:i + N, :2] - ref_traj[i:i + N, :2], axis=1) ** 2 +
             angle_wrap(actual_traj[i:i + N, 2] - ref_traj[i:i + N, 2]) ** 2 for i in range(len(ref_traj) - N)]

    xcost.reverse()
    cost = []
    for i in range(len(ref_traj) - N):
        tot = list(accumulate(xcost[i], lambda x, y: x * gamma + y))
        cost.append(tot[-1])
    cost.reverse()
    return np.vstack(cost), np.vstack(input_traj)


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






