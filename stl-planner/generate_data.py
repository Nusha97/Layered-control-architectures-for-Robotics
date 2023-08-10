"""
SYNOPSIS
    A simple trajectory generator code for nonlinear dynamical systems

DESCRIPTION
    Generates multiple trajectories for the given nonlinear system using ILQR
    based on specified parameters. Currently, no constraints on inputs have been
    implemented. Provided implementation is used for the unicycle model. This can
    be easily modified to test other nonlinear systems in simulation.

AUTHOR
    Anusha Srikanthan <sanusha@seas.upenn.edu>

VERSION
    0.0
"""

import functools

import jax.numpy as np
import numpy as onp
import jax

from trajax import optimizers
from trajax.integrators import euler
from trajax.integrators import rk4

import pickle
from helper_functions import angle_wrap, compute_rdot

gamma = 1


def save_object(obj, filename):
    """
    Save object as a pickle file
    :param obj: Data to be saved
    :param filename: file path
    :return: None
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(str):
    """
    Load object given a file path
    :param str: file path
    :return: file object
    """
    with open(str, 'rb') as handle:
        return pickle.load(handle)


class ILQR():
    """
    Apply iterative LQR from trajax on any specified dynamical system
    """

    def __init__(self, dynamics, maxiter=1000):
        self.maxiter = maxiter
        self.dynamics = dynamics
        self.constraints_threshold = 1.0e-1
        self.goal = None
        self.horizon = None

    def discretize(self, type):
        if type == 'euler':
            return euler(self.dynamics, dt=0.01)
        else:
            return rk4(self.dynamics, dt=0.01)

    def apply_ilqr(self, x0, U0, goal, wp, ts, maxiter=None, true_params=(100, 1.0, 0.1)):
        """
        Function to execute ilqr from trajax to generate reference trajectories
        :return:
        :param x0: initial state of the system
        :param U0: initial input trajectory
        :param goal: goal states of the system
        :param wp: waypoints
        :param ts: waypoint times
        :param maxiter: number of iterations to run ilqr
        :param true_params: Specify cost weights for the ilqr cost function
        :return: X, U, total_iter, dynamics
        """
        dynamics = self.discretize('rk4')

        if maxiter:
            self.maxiter = maxiter

        self.goal = goal

        self.horizon = U0.shape[0]

        key = jax.random.PRNGKey(75493)
        stage_cost = onp.zeros((self.horizon, 2, 2))
        for i in range(self.horizon):
            if i == ts[1]:
                stage_cost[i, :, :] = np.array([[10, 0], [0, 10]])
            else:
                stage_cost[i, :, :] = np.eye(2)
        true_param = (20 * np.eye(2), np.array(stage_cost), 0.00001 * np.eye(3))

        def cost(params, state, action, t):
            """
            Cost function for ilqr - designed assuming the state (x, r) and action (rdot)
            :param params: weight matrices and PD gain constants
            :param state: x, r
            :param action: rdot
            :param t: current time step
            :return: cost
            """
            final_weight, stage_weight, cost_weight = params

            st_err = state[3:5] - state[0:2]
            st_cost = np.matmul(np.matmul(st_err, np.squeeze(stage_weight[t])), st_err)
            wp_err = state[0:2] - np.array(wp[t, :])
            wp_cost = np.matmul(np.matmul(wp_err, np.squeeze(stage_weight[t])), wp_err)
            state_cost = np.where(t == ts[1], st_cost + wp_cost, st_cost)
            action_cost = np.matmul(np.matmul(np.squeeze(action), cost_weight), np.squeeze(action))
            terminal_err = state[0:2] - self.goal
            terminal_cost = np.matmul(np.matmul(terminal_err, final_weight), terminal_err)

            return np.where(t == self.horizon, terminal_cost, state_cost + action_cost)

        def equality_constraint(x, u, t):
            del u
            # maximum constraint dimension across time steps
            dim = 2

            def goal_constraint(x):
                err = x[0:2] - self.goal
                return err
            return np.where(t == self.horizon, goal_constraint(x), np.zeros(dim))

        X, U, _, _, _, _, _, _, _, _, _, total_iter = optimizers.constrained_ilqr(functools.partial(cost, true_param), dynamics, x0,
                                    U0, equality_constraint=equality_constraint, constraints_threshold=self.constraints_threshold, maxiter_al=self.maxiter)
        return X, U, total_iter, dynamics


def generate_polynomial_coeffs(start, end, T, order):
    """
    Generates a polynomial trajectory from start to end over time T
    :param: start: start state
    :param: end: end state
    :param: T: total time
    :param: order: order of the polynomial
    :return: coeffs
    """
    # Define the time vector
    t = onp.linspace(0, 1, T)
    # Solve for the polynomial coefficients
    coeffs = onp.polyfit(t, t * (end - start) + start, order)
    return coeffs


def gen_uni_training_data(lqr_obj, num_iter, file_path, state_dim, inp_dim, goals=None, inits=None):
    """
    Generate trajectories for training data
    :param dynamics: Dynamics function to be passed to ILQR
    :param goals: Set of goals generated using random generator
    :param inits: Set of initial conditions to be tested on
    :return: xtraj, rtraj, rdottraj, costs
    """

    horizon = 10
    dt = 0.01

    xtraj = []
    rtraj = []
    rdottraj = []
    costs = []

    onp.random.seed(10)

    if goals == None:
        goals = onp.random.uniform(1, 3, (num_iter, 2))
        print(goals)

    if inits == None:
        init_xy = onp.random.uniform(0, 2, (num_iter, 2))
        init_theta = onp.random.uniform(0, onp.pi, (num_iter, ))
        init = onp.append(init_xy, init_theta)
        inits = onp.reshape(init, (num_iter, int(state_dim/2)), order='F')
        print(inits)

    ts = np.linspace(0, horizon-1, 3, dtype=int)

    for j in range(num_iter):
        wp = onp.zeros((horizon, 2))
        wp[ts[0], :] = inits[j, :2]
        wp[ts[1], :] = inits[j, :2]*2/3 + goals[j, :]/2
        wp[ts[2], :] = goals[j, :]

        tk_cost = 100000
        it = 2
        while tk_cost/horizon >= 1:
            if it > 1000:
                break
            x0 = np.append(inits[j, :], inits[j, :])
            U0 = np.zeros(shape=(horizon-1, inp_dim))
            g = goals[j, :]
            x, u, t_iter, dynamics = lqr_obj.apply_ilqr(x0, U0, g, np.array(wp), ts, it)
            it *= 5
            r_int = x[:, 3:]
            tk_cost = onp.linalg.norm(x[:, 0:3] - r_int)

            if tk_cost > 500:
                continue

            xtraj.append(x[:, 0:3])
            rtraj.append(r_int)
            rdottraj.append(u)
            costs.append(tk_cost)

            # save_object([xtraj, rtraj, rdottraj, costs], file_path)

    return xtraj, rtraj, rdottraj, costs, dynamics


def unicycle_K(x, u, t):
    """
    Unicycle system controlled by a controller with proportional constant Kp and derivative constant Kd
    :param x: 6 dim-states of the dynamical system (x, y, theta, r)
    :param u: Control input (v, w)
    :param t: time step to evaluate
    :return: xdot
    """
    del t
    px, py, theta = x[0:3]
    theta = angle_wrap(theta)

    x_dev = x[0:3] - x[3:]

    F = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    Kp = 5 * np.array([[2, 1, 0], [0, 1, 3]])
    Kd = np.linalg.pinv(F)

    return np.append(np.matmul(F, np.matmul(Kd, np.squeeze(u)) + np.matmul(Kp, x_dev)), np.squeeze(u))


def forward_simulate(dynamics, x0, r, u, horizon):
    """
    System rollout computed by forward simulating the discretized dynamics of the closed-loop unicycle model
    :param dynamics: discretized dynamics of unicycle
    :param x0: initial state
    :param r: reference trajectory
    :param u: rdot control input
    :param horizon: length of the trajectory
    :return: cost, x (3 dim-state)
    """
    if u is None:
        u = compute_rdot(r, 0.01)
    x = onp.zeros((horizon, 6))
    x[0, :3] = x0
    x[:, 3:] = r
    for i in range(horizon-1):
        state_traj = dynamics(x[i, :], u[i, :], i)
        x[i+1, :] = state_traj

    cost = onp.sum(onp.linalg.norm(x[:, :2] - r[:, :2], axis=1) ** 2 + angle_wrap(x[:, 2] - r[:, 2]) ** 2)

    return cost, x[:, :3]


