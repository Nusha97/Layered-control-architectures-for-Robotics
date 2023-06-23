"""
Script to test the ADMM algorithm for a linear system
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


def V(z, P, p, c):
    return z.T @ P @ z + 2 * p.T @ z + c


def riccati_recursion(A, B, Q, q, R, N):
    P = [None] * (N + 1)
    p = [None] * (N + 1)
    K = [None] * N
    M = [None] * N
    c = [None] * (N + 1)
    v = [None] * N

    P[N] = Q
    p[N] = q
    c[N] = 0

    Mu = []

    for t in range(N - 1, -1, -1):
        K[t] = np.linalg.inv(B.T @ P[t + 1] @ B + R) @ B.T @ P[t + 1] @ A
        v[t] = np.linalg.inv(B.T @ P[t + 1] @ B + R) @ B.T @ p[t + 1]
        M[t] = K[t].T @ R @ np.linalg.inv(B.T @ P[t + 1] @ B + R) @ B.T + (A - B @ K[t]).T @ (
                    np.eye(A.shape[0]) - P[t + 1] @ B @ np.linalg.inv(B.T @ P[t + 1] @ B + R) @ B.T)
        P[t] = Q + A.T @ P[t + 1] @ A - A.T @ P[t + 1] @ B @ K[t]
        p[t] = q + K[t].T @ R @ v[t] + (A - B @ K[t]).T @ (p[t + 1] - P[t + 1] @ B @ v[t])
        c[t] = -p[t + 1].T @ B @ np.linalg.inv(B.T @ P[t + 1] @ B + R) @ B.T @ p[t + 1] + c[t + 1]

    return P, p, c, K, M, v


def planning(P0, Psi, x0, Nu, n, N, Q):
    r = cp.Variable(shape=(n, N + 1))
    util_cost = [cp.quad_form(r[:, i], Q) for i in range(N + 1)]
    z0 = np.append(x0, r)

    P_bar = P0 + Nu @ np.linalg.inv(Psi) @ Nu.T
    print("P_bar", P_bar.shape)
    track_cost = cp.quad_form(z0, P_bar)

    obj = cp.Minimize(cp.sum[util_cost, track_cost])

    constr = []
    constr.append(r[:, 0] == x0)

    prob = cp.Problem(obj, constr)

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)

    plt.figure()
    plt.plot(r[0].value, r[1].value)

    return r.value


def reference_tracking(A, B, Q, R, N, F, r, n, m, mu, x0):
    # Using cvxpy

    x = cp.Variable(shape=(n, N + 1))
    # z = cp.Variable(shape=(n * (N+2), N+1))
    u = cp.Variable(shape=(m, N))

    # z = np.zeros([n * (N+2), N+1])

    constr = []

    stage_cost = [cp.quad_form(x[:, i] - r[:, i], Q) + cp.quad_form(u[:, i], R) for i in range(N)]
    term_cost = cp.quad_form(x[:, N] - r[:, N], Q)

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
    plt.plot(x[0].value, x[1].value, 'r--')
    plt.plot(r[0], r[1])

    # Using Riccati

    E1 = np.vstack([np.eye(n), np.zeros([n * N, n])])
    E2 = np.vstack([np.zeros([n, n]), np.eye(n), np.zeros([n * (N - 1), n])])
    M = np.eye(n * (N + 1), n * (N + 1), 1)

    Q_bar = np.block(
        [[Q, np.zeros([n, (N + 1) * n])], [np.zeros([(N + 1) * n, n]), np.zeros([(N + 1) * n, (N + 1) * n])]])
    A_bar = np.block([[A, E2.T - A @ E1.T], [np.zeros([n * (N + 1), n]), M]])
    B_bar = np.vstack([B, np.zeros([n * (N + 1), m])])

    q = np.zeros([n * (N + 2), N + 1])
    for i in range(N + 1):
        q[:, i] = F.T @ mu[:, 0]

    P, p, c, K, M, v = riccati_recursion(A_bar, B_bar, Q_bar, q, R, N)

    z0 = np.append(x0, r.ravel(order='F'))

    # TODO: Fix dimensions

    cost = V(z0, P[0], p[0].T[0], c[0][0, 0])
    print("Reference Riccati cost", cost)

    Nu_mat = []
    for i in range(N):
        if i > 0:
            Nu_mat.append(Nu_mat[-1] @ M[i])
        else:
            Nu_mat.append(M[i])

    Nu = np.block([Nu_mat[i] @ F.T for i in range(N)])

    Psi_mat = []

    for i in range(N):
        Psi_mat.append(
            F @ Nu_mat[i].T @ B_bar @ np.linalg.inv(R + B_bar.T @ P[i + 1] @ B_bar) @ B_bar.T @ Nu_mat[i] @ F.T)

    # TODO: Change this
    Psi = block_diag(Psi_mat[0], Psi_mat[1], Psi_mat[2], Psi_mat[3], Psi_mat[4], Psi_mat[5], Psi_mat[6], Psi_mat[7],
                     Psi_mat[8], Psi_mat[9])

    ## Evolution of trajectory from layering
    u_lqr = np.zeros([m, N])
    x_lqr = np.zeros([n, N + 1])
    z_lqr = np.zeros([n * (N + 2), N + 1])

    x_lqr[:, 0] = x0

    for i in range(N + 1):
        z_lqr[:n, i] = x_lqr[:, i] - r[:, i]
        z_lqr[n:(N - i + 2) * n, i] = r[:, i:].ravel(order='F')

    for i in range(N):
        u_lqr[:, i] = -K[N - i - 1] @ z_lqr[:, i] - v[N - i - 1][:, 0]
        z_lqr[:, i + 1] = A_bar @ z_lqr[:, i] + B_bar @ u_lqr[:, i]

    plt.figure()
    plt.plot(z_lqr[0, :], z_lqr[1, :], label="ref tracking")
    plt.legend()

    return P, p, c, Nu, Psi, z_lqr, u_lqr


def main():
    N = 10

    n = 2
    m = 2

    A = np.array([[1, 1], [0, 1]])
    B = np.eye(m)

    Q = np.array([[10, 0], [0, 1]])
    R = np.eye(m)

    q = np.ones(2)

    x0 = np.array([10, 10])

    P, p, c, K, M, v = riccati_recursion(A, B, Q, q, R, N)

    print(P)

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

    x = cp.Variable(shape=(n, N + 1))
    u = cp.Variable(shape=(m, N))

    constr = []

    obj = cp.Minimize(cp.sum([cp.quad_form(x[:, i], Q) + cp.quad_form(u[:, i], R) + 2 * q @ x[:, i] for i in range(N)]))
    constr.append(x[:, 0] == x0)

    print(constr)

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

    F = np.hstack([np.eye(n), np.zeros([n, n * (N + 1)])])
    mu = 0.1 * np.ones([n, N + 1])

    # reference_tracking(A, B, Q, R, N, F, np.zeros(x.shape), n, m, mu, x0)

    # ADMM
    # not_converged = True

    # while not_converged:
    P, p, c, Nu, Psi, x, u = reference_tracking(A, B, Q, R, N, F, np.zeros(x.shape), n, m, mu, x0)
    r = planning(P, Psi, x0, Nu, n, N, Q)
    # mu --> updates


if __name__ == '__main__':
    main()