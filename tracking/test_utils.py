##############################################################################
# Fengjun Yang, 2022
# Utility functions for testing the learned value and tracking controllers
##############################################################################

import numpy as np


def random_env(p, q, Anorm=0.95):
    #TODO: make sure that the system is controllable
    """ Generate a random linear dynamical system with specified dimensions.
    Input:
        - p:        Integer, state dimension
        - q:        Integer, control dimension
        - Anorm:    Float, spectral radius of A
    Output:
        - A:        np.array(p,p), dynamics matrix
        - B:        np.array(p,q), controls matrix
    """
    A = np.random.randn(p, p)
    A = A / np.abs(np.linalg.eigvals(A)).max() * Anorm
    B = np.random.randn(p, q)
    return A, B


def random_controller(q):
    """ Returns a random controller
    Input:
        - q:        Integer, control dimension
    Return:
        - ctrl:     function(p -> q), outputs a random control action
    """
    return lambda x: np.random.random(q) - 0.5


def zero_controller(q):
    """ Returns a zero controller
    Input:
        - q:        Integer, control dimension
    Return:
        - ctrl:     function(p -> q), outputs zero control
    """
    return lambda x: np.zeros(q)


def linear_feedback_controller(K):
    """ Returns a linear feedback controller
    Input:
        - K:        np.array, feedback gain
    Return:
        - ctrl:     function(p -> q), outputs zero control
    """
    return lambda x: np.dot(K, x)

class TVcontroller():
    def __init__(self, K):
        self.K = K

    def ctrl(self, x):
        K = self.K.pop(0)
        return K @ x

def sample_traj(A, B, Q, R, ctrl, T, x0=None, sigma=1):
    """ Given an environment, sample a trajectory of T time steps
    Input:
        - A, B:     dynamics of the system
        - Q, R:     cost structure
        - ctrl:        controller
        - T:        length of trajectory
        - x0:       initial state. If None, it will be sampled at random
    Return:
        - xtraj:    np.array(T+1, p), trajectory of state
        - utraj:    np.array(T, q), trajectory of control
        - rtraj:    np.array(T), cost at each step
    """
    # Generate x0 if x0 is None
    if x0 is None:
        x0 = np.random.randn(A.shape[0])

    # Initialize variables
    p, q = B.shape
    xtraj = np.zeros((T+1, p))
    utraj = np.zeros((T, q))
    rtraj = np.zeros(T)

    # Simulate forward
    x = x0.copy()
    xtraj[0] = x
    for t in range(T):
        u = ctrl(x)
        x_ = A @ x + B @ u + sigma * np.random.randn(p)
        r = np.dot(Q @ x, x) + np.dot(R @ u, u)
        utraj[t] = u
        rtraj[t] = r
        xtraj[t+1] = x_
        x = x_
    return xtraj, utraj, rtraj

def sample_ref_traj(p, T, order=4):
    # TODO: move this into a tracking_utils.py file since it's a shared
    # functionality for all tracking controllers
    # TODO: sample random trajectories with given start and goal points
    """ Samples a reference trajectory to track. Does so by randomly sampling a
    polynomial and evaluating at points evenly spread on the unit interval.
    """
    coeffs = np.random.randn(p, order)
    coeffs = coeffs / coeffs.sum(1)[:, None]
    points = np.array([np.linspace(0,1,T) ** (i+1) for i in range(order)])
    return coeffs @ points

def lqr_vi(A, B, Q, R, T, gamma=1):
    """ Solves lqr for a finite time horizon
    """
    P = Q
    K = [-np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A]
    for t in range(T-1):
        P = Q + K[0].T @ R @ K[0] + gamma * (A + B@K[0]).T @ P @ (A + B@K[0])
        K = [-np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A] + K
    return P, K

def relerr(A, Ahat):
    """ computes relative error of two matrices in terms of frobenius norm """
    return np.linalg.norm(A-Ahat, 'fro') / np.linalg.norm(A, 'fro')
