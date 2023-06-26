"""
Script to test the ADMM algorithm for a linear system
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import sys
sys.path.insert(0, "/home/anusha/Research/Layered-architecture-quadrotor-control/Simulations")

from trajgen import quadratic


def V(z, P, p, c):
    return z.T @ P @ z + 2 * p.T @ z + c


def V_cost(x0, r, P, p, c, n):
    P_11 = P[:n, :n]
    P_12 = P[:n, n:]
    P_22 = P[n:, n:]

    return cp.quad_form(x0, P_11) + cp.quad_form(r, P_22) + 2 * x0 @ (P_12 @ r) + p[n:].T @ r + p[:n].T @ x0 + c


def riccati_recursion(A, B, Q, q, R, N):
    P = [None] * (N + 1)
    p = [None] * (N + 1)
    K = [None] * N
    M = [None] * N
    c = [None] * (N + 1)
    v = [None] * N

    P[N] = Q[N]
    p[N] = q[N]
    c[N] = 0

    Mu = []

    for t in range(N - 1, -1, -1):
        K[t] = np.linalg.inv(B.T @ P[t + 1] @ B + R) @ B.T @ P[t + 1] @ A
        v[t] = np.linalg.inv(B.T @ P[t + 1] @ B + R) @ B.T @ p[t + 1]
        M[t] = K[t].T @ R @ np.linalg.inv(B.T @ P[t + 1] @ B + R) @ B.T + (A - B @ K[t]).T @ (
                    np.eye(A.shape[0]) - P[t + 1] @ B @ np.linalg.inv(B.T @ P[t + 1] @ B + R) @ B.T)
        P[t] = Q[t] + A.T @ P[t + 1] @ A - A.T @ P[t + 1] @ B @ K[t]
        p[t] = q[t] + K[t].T @ R @ v[t] + (A - B @ K[t]).T @ (p[t + 1] - P[t + 1] @ B @ v[t])
        c[t] = -p[t + 1].T @ B @ np.linalg.inv(B.T @ P[t + 1] @ B + R) @ B.T @ p[t + 1] + c[t + 1]

    return P, p, c, K, M, v


def layered_planning(x0, P, p, c, n, N, Q):
    r = cp.Variable(shape=(n * (N + 1)))

    track_cost = V_cost(x0, r, P, p, c, n)

    waypoints = [[10, 10], [5, 5], [0, 0]]
    ts = np.linspace(0, 1, len(waypoints))
    order = 5

    # objective, constr, ref, coeff = quadratic.min_jerk_setup(waypoints, ts, order, n, N+1)

    # util_cost = objective
    util_cost = cp.quad_form(r, np.eye(r.shape[0]))

    obj = cp.Minimize(util_cost + track_cost)

    constr = []
    constr.append(r[:n] == x0)

    prob = cp.Problem(obj, constr)
    prob.solve()

    z0 = np.append(x0, r.value)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution z0 is")
    print(r.value)
    print("A dual solution corresponding to the inequality constraints is")
    # print(prob.constraints[0].dual_value)

    plt.figure()
    plt.plot(z0[n::2], z0[n + 1::2], label="layered ref")
    plt.close()

    return z0


def planning(P0, Psi, x0, Nu, n, N, Q):
    P_bar = P0 + Nu @ np.linalg.inv(Psi) @ Nu.T

    # P_bar = P0

    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) >= 0)

    print(is_pos_def(P_bar))

    r = cp.Variable(shape=(n * (N + 1)))

    P_11 = P_bar[:n, :n]
    P_12 = P_bar[:n, n:]
    P_22 = P_bar[n:, n:]

    util_cost = cp.quad_form(r, np.eye(r.shape[0]))
    track_cost = cp.quad_form(r, P_22) + cp.quad_form(x0, P_11) + 2 * x0 @ (P_12 @ r)

    obj = cp.Minimize(util_cost + track_cost)

    constr = []
    constr.append(r[:n] == x0)

    prob = cp.Problem(obj, constr)
    prob.solve()

    z0 = np.append(x0, r.value)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution z0 is")
    print(r.value)
    print("A dual solution corresponding to the inequality constraints is")
    # print(prob.constraints[0].dual_value)

    # plt.figure()
    # plt.plot(z0[n::2], z0[n+1::2], label="ref lqr")
    # plt.close()

    return z0


def reference_tracking(A, B, Q, R, N, F, r, n, m, mu, x0):
    # Using cvxpy

    x = cp.Variable(shape=(n, N + 1))
    u = cp.Variable(shape=(m, N))

    constr = []

    stage_cost = [cp.quad_form(x[:, i] - r[:, i], Q[i]) + cp.quad_form(u[:, i], R) for i in range(N)]
    term_cost = cp.quad_form(x[:, N] - r[:, N], Q[N])

    stage_cost.append(term_cost)
    obj = cp.Minimize(cp.sum(stage_cost))

    constr.append(x[:, 0] == x0)

    for i in range(N - 1):
        constr.append(x[:, i + 1] == A @ x[:, i] + B @ u[:, i])

    prob = cp.Problem(obj, constr)

    prob.solve()

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)

    plt.figure()
    # plt.plot(x[0].value, x[1].value, 'r--', label="lqr cvxpy")
    plt.plot(x[0].value, 'r--', label="lqr cvxpy")
    # plt.plot(r[0], r[1], label="ref cvxpy")
    plt.plot(r[0], label="ref cvxpy")
    plt.legend()

    # Using Riccati
    # TODO: Need to figure out why the Riccati solution is mildly different cvxpy

    E1 = np.vstack([np.eye(n), np.zeros([n * N, n])])
    E2 = np.vstack([np.zeros([n, n]), np.eye(n), np.zeros([n * (N - 1), n])])
    M = np.eye(n * (N + 1), n * (N + 1), 1)

    Q_bar = np.block(
        [[Q[0], np.zeros([n, (N + 1) * n])], [np.zeros([(N + 1) * n, n]), np.zeros([(N + 1) * n, (N + 1) * n])]])
    A_bar = np.block([[A, E2.T - A @ E1.T], [np.zeros([n * (N + 1), n]), M]])
    B_bar = np.vstack([B, np.zeros([n * (N + 1), m])])

    q = []
    for i in range(N + 1):
        q.append(F.T @ mu[:, i])

    Q_bar_list = [Q_bar for i in range(N + 1)]
    P, p, c, K, M, v = riccati_recursion(A_bar, B_bar, Q_bar_list, q, R, N)

    z0 = np.append(x0, r.ravel(order='F'))

    cost = V(z0, P[0], p[0], c[0])
    print("Reference Riccati cost", cost)

    Nu_mat = []
    for i in range(N + 1):
        if i > 0:
            Nu_mat.append(Nu_mat[-1] @ M[i - 1])
        else:
            Nu_mat.append(np.eye(q[i].shape[0]))

    Nu = np.block([Nu_mat[i] @ F.T for i in range(N + 1)])

    Psi_mat = []

    for i in range(N + 1):
        Psi_mat.append(F @ Nu_mat[i].T @ B_bar @ np.linalg.inv(R + B_bar.T @ P[i] @ B_bar) @ B_bar.T @ Nu_mat[i] @ F.T)

    # TODO: Change the computation of Psi
    Psi = block_diag(Psi_mat[0], Psi_mat[1], Psi_mat[2], Psi_mat[3], Psi_mat[4], Psi_mat[5], Psi_mat[6], Psi_mat[7],
                     Psi_mat[8], Psi_mat[9], Psi_mat[10])

    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) >= 0)

    print("Pos def", is_pos_def(Psi))

    ## Evolution of trajectory from layering
    u_lqr = np.zeros([m, N])
    x_lqr = np.zeros([n, N + 1])
    z_lqr = np.zeros([n * (N + 2), N + 1])

    x_lqr[:, 0] = x0

    for i in range(N + 1):
        z_lqr[:n, i] = x_lqr[:, i] - r[:, i]
        z_lqr[n:(N - i + 2) * n, i] = r[:, i:].ravel(order='F')

    for i in range(N):
        u_lqr[:, i] = -K[N - i - 1] @ z_lqr[:, i] - v[N - i - 1]
        z_lqr[:, i + 1] = A_bar @ z_lqr[:, i] + B_bar @ u_lqr[:, i]

    x_lqr = z_lqr[:n, :] + r

    plt.figure()
    # plt.plot(z_lqr[0, :], z_lqr[1, :], label="ref tracking")
    plt.plot(z_lqr[0, :] + r[0, :], label="lqr riccati")
    # plt.plot(r[0, :], r[1, :], label="ref")
    plt.plot(r[0, :], label="ref riccati")
    plt.legend()
    # plt.close()

    return P, p, c, Nu, Psi, x_lqr, u_lqr


def main():
    N = 10

    n = 2
    m = 2

    A = np.array([[1, 1], [0, 1]])
    B = np.eye(m)

    Q = np.array([[10, 0], [0, 1]])
    Q_list = [Q for i in range(N + 1)]

    R = np.eye(m)

    q = np.ones(2)
    q_list = [q for i in range(N + 1)]

    x0 = np.array([10, 10])

    P, p, c, K, M, v = riccati_recursion(A, B, Q_list, q_list, R, N)

    cost = V(x0, P[0], p[0], c[0])

    print("Riccati cost", cost)

    ## Evolution of trajectory from layering
    u_lqr = np.zeros([m, N])
    x_lqr = np.zeros([n, N + 1])

    x_lqr[:, 0] = x0

    for i in range(N):
        u_lqr[:, i] = -K[N - i - 1] @ x_lqr[:, i] - v[N - i - 1]
        x_lqr[:, i + 1] = A @ x_lqr[:, i] + B @ u_lqr[:, i]

    plt.figure()
    plt.plot(x_lqr[0, :], x_lqr[1, :])
    plt.close()

    x = cp.Variable(shape=(n, N + 1))
    u = cp.Variable(shape=(m, N))

    constr = []

    obj = cp.Minimize(cp.sum([cp.quad_form(x[:, i], Q) + cp.quad_form(u[:, i], R) + 2 * q @ x[:, i] for i in range(N)]))
    constr.append(x[:, 0] == x0)

    for i in range(N - 1):
        constr.append(x[:, i + 1] == A @ x[:, i] + B @ u[:, i])

    prob = cp.Problem(obj, constr)

    prob.solve()

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)

    plt.figure()
    plt.plot(x[0].value, x[1].value)
    plt.close()

    F = np.hstack([np.eye(n), np.zeros([n, n * (N + 1)])])

    # TODO: How to set the optimal mu value?
    mu = 0.1 * np.ones([n, N + 1])

    # reference_tracking(A, B, Q, R, N, F, np.zeros(x.shape), n, m, mu, x0)

    P, p, c, Nu, Psi, x, u = reference_tracking(A, B, Q_list, R, N, F, np.zeros(x.shape), n, m, mu, x0)
    Psi = np.eye(Psi.shape[0])
    r = planning(P[0], Psi, x0, Nu, n, N, Q)
    P, p, c, Nu, Psi, x, u = reference_tracking(A, B, Q_list, R, N, F, np.zeros(x.shape), n, m, mu, x0)

    z0 = layered_planning(x0, P[0], p[0], c[0], n, N, Q)
    # r = planning(P[0], Psi, x0, Nu, n, N, Q)

    r = np.reshape(z0[n:], (n, N + 1))
    print(r)
    P, p, c, Nu, Psi, x, u = reference_tracking(A, B, Q_list, R, N, F, r, n, m, mu, x0)

    print(r)
    print(x)

    # ADMM
    not_converged = True

    # while not_converged:
    for i in range(5):
        z0 = layered_planning(x0, P[0], p[0], c[0], n, N, Q)
        # z0 = planning(P[0], Psi, x0, Nu, n, N, Q)
        r = np.reshape(z0[n:], (n, N + 1))
        P, p, c, Nu, Psi, x, u = reference_tracking(A, B, Q_list, R, N, F, r, n, m, mu, x0)

        print(r.shape)
        print(x.shape)
        mu = mu + r - x

        print("Deviation", np.linalg.norm(r - x))
        if np.linalg.norm(r - x) / 5 * N < 1:
            print("Is it time to break?")
            break
        #    not_converged = False


if __name__ == '__main__':
    main()
