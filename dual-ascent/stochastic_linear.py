"""
    ADMM for a linear system with disturbances
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def open_loop(n, m, T, x0, s, q, A, B, H, R, w):
    ## CVXPY baseline solution

    x_opt = cp.Variable((n, T))
    u_opt = cp.Variable((m, T - 1))

    # err  = x_opt[0,:] - s
    err = x_opt - np.vstack([s, q])
    utility_cost = cp.sum_squares(err)
    ctrl_cost = cp.sum([cp.quad_form(u_opt[:, t], R) for t in np.arange(T - 1)])

    dynamics_constraints = [x_opt[:, 0] == x0]

    for t in np.arange(T - 1):
        dynamics_constraints.append(x_opt[:, t + 1] == A @ x_opt[:, t] + B @ u_opt[:, t] + H @ w[:, t])

    prob = cp.Problem(cp.Minimize(utility_cost + ctrl_cost), dynamics_constraints)
    prob.solve()
    print(prob.value)

    plt.figure()
    # plt.plot(np.arange(T), s, linestyle="solid", linewidth=4)
    plt.plot(s, q, linestyle="solid", linewidth=4)
    plt.plot(x_opt.value[0, :], x_opt.value[1, :], linestyle="solid", linewidth=4)
    plt.legend(["ref", "ocp"], loc="lower left")
    plt.title("State evolution from optimal control with process noise")
    plt.xlabel("state 1")
    plt.ylabel("state 2")
    plt.show()


def admm(n, m, T, x0, s, q, A, B, H, R, w):
    rho = 25
    # Let's setup each suproblem as parameteric problems so we can call them in a loop
    # r-subproblem
    r = cp.Variable((n, T))
    xk = cp.Parameter((n, T))
    uk = cp.Parameter((m, T - 1))
    vk = cp.Parameter((n, T))
    rhok = cp.Parameter(nonneg=True)

    rhok.value = rho
    xk.value = np.zeros((n, T))
    uk.value = np.zeros((m, T - 1))
    vk.value = np.zeros((n, T))

    # err = r[0,:] - s
    err = r - np.vstack([s, q])
    utility_cost = cp.sum_squares(err)
    admm_cost = rhok * cp.sum_squares(r - xk + vk)

    r_subprob = cp.Problem(cp.Minimize(utility_cost + admm_cost), [r[:, 0] == x0])

    # (x,u) subproblem

    rk = cp.Parameter((n, T))
    rk.value = np.zeros((n, T))
    rk.value[0, :] = s
    x = cp.Variable((n, T))
    u = cp.Variable((m, T - 1))
    vk = cp.Parameter((n, T))
    vk.value = np.zeros((n, T))
    rhok = cp.Parameter(nonneg=True)
    rhok.value = rho

    ctrl_cost = cp.sum([cp.quad_form(u[:, t], R) for t in np.arange(T - 1)])
    admm_cost = rhok * cp.sum_squares(rk - x + vk)

    dynamics_constraints = [x[:, 0] == x0]

    for t in np.arange(T - 1):
        dynamics_constraints.append(x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + H @ w[:, t])

    xu_subprob = cp.Problem(cp.Minimize(ctrl_cost + admm_cost), dynamics_constraints)

    # run ADMM algorithms
    K = 100
    res = []
    des = []

    xk.value = np.zeros((n, T))
    uk.value = np.zeros((m, T - 1))
    vk.value = np.zeros((n, T))
    rhok.value = 50

    for k in np.arange(K):
        # update r
        r_subprob.solve()
        # print("Is DPP? ", r_suprob.is_dcp(dpp=True))

        # update x u
        rk.value = r.value
        xu_subprob.solve()
        # print("Is DPP? ", xu_subprob.is_dcp(dpp=True))

        # compute residuals
        sxk = rhok.value * (xk.value - x.value).flatten()
        suk = rhok.value * (uk.value - u.value).flatten()
        dual_res_norm = np.linalg.norm(np.hstack([sxk, suk]))
        pr_res_norm = np.linalg.norm(rk.value - xk.value)

        # update rhok and rescale vk
        if pr_res_norm > 10 * dual_res_norm:
            rhok.value = 2 * rhok.value
            vk.value = vk.value / 2
        elif dual_res_norm > 10 * pr_res_norm:
            rhok.value = rhok.value / 2
            vk.value = vk.value * 2

        # update v
        xk.value = x.value
        uk.value = u.value
        vk.value = vk.value + rk.value - xk.value
        residual = np.vstack([s, q]) - xk.value
        des.append(np.trace(residual.T @ residual))
        residual = rk.value - xk.value
        res.append(np.trace(residual.T @ residual))

    plt.figure()
    plt.plot(s, q, linestyle="solid", linewidth=4)
    plt.plot(r.value[0, :], r.value[1, :], linestyle="solid", linewidth=4)
    plt.plot(x.value[0, :], x.value[1, :], linestyle="dashed", linewidth=4)
    plt.legend(["init_ref", "admm_ref", "admm_state"], loc="lower left")
    plt.title("State evolution from admm")
    plt.xlabel("state 1")
    plt.ylabel("state 2")
    plt.show()


def main():
    # Set up problem parameters
    n = 2
    m = 2
    T = 20

    # n-dim chain of integrators
    A = np.diag(np.ones(n)) + np.diag(np.ones(n - 1), 1)

    # actuators enter through the bottom of the chain
    B = np.zeros((n, m))
    B[-m:, :] = np.eye(m)

    # Noise matrix
    H = np.eye(n)

    # Disturbances
    np.random.seed(100)
    rng = np.random.default_rng()
    w = rng.standard_normal((n, T))

    # C(x) will be to have x_1 track a sinuisoidal trajectory s(omega * t)
    t = np.arange(T)
    omega = 0.5
    s = 0.5 * np.sin(omega * t)
    q = 0.5 * np.cos(omega * t)

    # tracking costs
    Q = np.eye(n)
    R = 0.1 * np.eye(m)

    # initial condition is x0 = 0
    x0 = np.zeros(n)

    # Baseline open loop
    open_loop(n, m, T, x0, s, q, A, B, H, R, w)

    # ADMM
    admm(n, m, T, x0, s, q, A, B, H, R, w)


if __name__ == '__main__':
    main()
