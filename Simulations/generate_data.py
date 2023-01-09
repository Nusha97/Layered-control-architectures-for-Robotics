"""
SYNOPSIS
    A simple trajectory generator code for nonlinear dynamical systems
DESCRIPTION

    Generates multiple trajectories for the given nonlinear system using ILQR
    based on specified parameters. Currently, no constraints on inputs have been
    implemented.
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

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

import matplotlib.pyplot as plt
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(str):
    with open(str, 'rb') as handle:
        return pickle.load(handle)


class ILQR():
    """
    Apply iterative LQR from trajax on any specified dynamical system
    """

    def __init__(self, dynamics, maxiter=100):
        self.maxiter = maxiter
        self.dynamics = dynamics


    def discretize(self, type):
        if type == 'euler':
            return euler(self.dynamics, dt=0.01)
        else:
            return rk4(self.dynamics, dt=0.01)


    def apply_ilqr(self, x0, U0, goal, maxiter=None, true_params=(100, 1.0, 0.1)):
        """
        Function to execute ilqr from trajax to generate reference trajectories
        :param true_params: Specify cost weights and gain constants of PD controller
        :return:
        """
        dynamics = self.discretize('rk4')
        m = x0.shape[0]

        if maxiter:
            self.maxiter = maxiter

        horizon = U0.shape[0]

        if m > 2:
            key = jax.random.PRNGKey(75493)
            true_params = (np.eye(3), np.array([[10, 0, 0], [0, 10, 0], [0, 0, 100]])* np.eye(3), 0.1 * np.eye(3))
            final_weight, stage_weight, cost_weight = true_params
        else:
            final_weight, stage_weight, cost_weight = true_params

        def cost(params, state, action, t):
            """
            Cost function should be designed assuming the state (x) and action (r, rdot)
            :param params: weight matrices and PD gain constants
            :param state: x, r
            :param action: r, rdot
            :param t: current time step
            :return: List of state, input, total_iter
            """
            final_weight, stage_weight, cost_weight = params

            # wrap to [-pi, pi]
            def angle_wrap(theta):
                return (theta + np.pi) % (2 * np.pi) - np.pi

            if m == 2:
                state_err = state[0] - np.squeeze(action[0])
                state_cost = stage_weight * state_err ** 2
                action_cost = np.squeeze(action[1]) ** 2 * cost_weight
                terminal_cost = final_weight * (state[0] - goal) ** 2
            else:
                state_err = state - np.squeeze(action[0:m])
                # val = state_err[m-1] - 2 * np.pi
                # angle_err = angle_wrap(state_err[m-1])
                # new_state_err = np.append(state_err[0:m-1], angle_err)
                # err = np.where(val, new_state_err, state_err)
                state_cost = np.matmul(np.matmul(state_err, stage_weight), state_err)
                # state_cost = np.matmul(np.matmul(state_err, stage_weight), state_err)
                action_cost = np.matmul(np.matmul(np.squeeze(action[m:]), cost_weight), np.squeeze(action[m:]))
                terminal_err = state - goal
                terminal_cost = np.matmul(np.matmul(terminal_err, final_weight), terminal_err)

            return np.where(t == horizon, terminal_cost, state_cost + action_cost)

        X, U, _, _, _, _, total_iter = optimizers.ilqr(functools.partial(cost, true_params), dynamics, x0,
                                                                    U0, self.maxiter)
        return X, U, total_iter


def gen_uni_training_data(lqr_obj, iter_list, num_iter, state_dim, inp_dim, goals=None, inits=None):
    """
    Generate trajectories for training data
    :param dynamics:
    :param iter_list:
    :param goals:
    :param inits:
    :return:
    """

    horizon = 100

    xtraj = []
    rtraj = []
    costs = []

    if goals == None:
        key = jax.random.PRNGKey(89731203)
        goal_xy = 19 * jax.random.normal(key, shape=(num_iter, 2))
        goal_theta = jax.random.uniform(key, shape=(num_iter,), minval=0, maxval=2*np.pi)
        goal = np.append(goal_xy, goal_theta)
        goals = np.reshape(goal, (num_iter, state_dim))

    if inits == None:
        key = jax.random.PRNGKey(95123459)
        init_xy = 28 * jax.random.normal(key, shape=(num_iter, 2))
        init_theta = jax.random.uniform(key, shape=(num_iter,), minval=0, maxval=2 * np.pi)
        init = np.append(init_xy, init_theta)
        inits = np.reshape(init, (num_iter, state_dim))

    for j in range(num_iter):
        tk_cost = 100
        it = 2
        while tk_cost/horizon >= 0.15:
            if it > 1000:
                break
            x0 = inits[j, :]
            U0 = np.zeros(shape=(horizon-1, inp_dim))
            if state_dim == 1:
                goal = goals[j, 0]
                U = np.append(np.array([0, inits[j, 1]]), U0)
            else:
                goal = goals[j, :]
                U = np.append(np.array([inits[j, :], np.zeros(state_dim)]), U0)
            x, u, t_iter = lqr_obj.apply_ilqr(x0, np.reshape(U, (horizon, inp_dim)), goal, it)
            tk_cost = np.linalg.norm(x[:-1, 0:state_dim]-u[:, 0:state_dim]) # + np.linalg.norm(x[-1, 0:state_dim] - goal)
            it *= 5

            if tk_cost > 500:
                continue

            xtraj.append(x)
            rtraj.append(u)
            costs.append(tk_cost)

            print(tk_cost)
            # plt.plot(x[:, 0], 'b-', u[:, 0], 'r--')
            # plt.show()

    return xtraj, rtraj, costs


def unicycle(x, u, t):
    """
    Unicycle system
    :param x:
    :param u:
    :param t:
    :return:
    """
    px, py, theta = x
    state_dim = 3
    F = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    Kp = np.array([[2, 0, 0], [0, 1, 0]])
    key = jax.random.PRNGKey(793)
    Kd = jax.random.uniform(key=key, shape=(2, 3))
    # x_dev = x[0:3] - np.squeeze(u[0:3])
    x_dev = x - np.squeeze(u[0:state_dim])

    v1 = np.matmul(np.matmul(F, np.linalg.inv(np.eye(2) - np.matmul(Kd, F))), np.matmul(Kp, x_dev) - np.matmul(Kd, np.squeeze(u[3:])))
    return np.array(v1)
