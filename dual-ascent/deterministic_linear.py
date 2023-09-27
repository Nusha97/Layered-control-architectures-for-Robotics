"""
ADMM applied to an optimal control problem
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def is_pos_def(x):
        # Check if matrix is psd
        return np.all(np.linalg.eigvals(x) >= 0)


def riccati_recursion(A, B, Q, q, R, N):

    P = [None] * (N+1)
    p = [None] * (N+1)
    K = [None] * N
    M = [None] * N
    c = [None] * (N+1)
    v = [None] * N

    P[N] = Q[N]
    p[N] = q[N]
    c[N] = q[N+1]

    for t in range(N-1, -1, -1):

        K[t] = np.linalg.inv(B.T @ P[t+1] @ B + R) @ B.T @ P[t+1] @ A
        v[t] = np.linalg.inv(B.T @ P[t+1] @ B + R) @ B.T @ p[t+1]
        M[t] = K[t].T @ R @ np.linalg.inv(B.T @ P[t+1] @ B + R) @ B.T + (A - B @ K[t]).T @ (np.eye(A.shape[0]) - P[t+1] @ B @ np.linalg.inv(B.T @ P[t+1] @ B + R) @ B.T)
        P[t] = Q[t] + A.T @ P[t+1] @ A - A.T @ P[t+1] @ B @ K[t]
        p[t] = q[t] + K[t].T @ R @ v[t] + (A - B @ K[t]).T @ (p[t+1] - P[t+1] @ B @ v[t])
        #c[t] = -p[t+1].T @ B @ np.linalg.inv(B.T @ P[t+1] @ B + R) @ B.T @ p[t+1] + c[t+1]
        c[t] = -v[t].T @ (B.T @ P[t+1] @ B + R) @ v[t] - 2 * p[t+1].T @ B @ v[t] + c[t+1]

    return P, p, c, K, M, v


def layered_planning(x0, x, v, n, N, rho):

    s = np.zeros((n * (N+1)))
    # s[0::2] = np.linspace(x0[0], 0, N+1)
    # s[1::2] = np.linspace(x0[1], 0, N+1)

    # C(x) will be to have x_1 track a sinuisoidal trajectory s(omega * t)
    t = np.arange(N+1)
    omega = 0.5
    s[0::2] = 2 * np.sin(omega * t)

    r = cp.Variable(shape=(n * (N+1)))

    err = x.reshape(n*(N+1), order='F') - r + v.reshape(n*(N+1), order='F')

    track_cost = (rho/2) * cp.sum_squares(err)


    util_cost = cp.quad_form(r-s, np.eye(r.shape[0]))

    obj = cp.Minimize(util_cost + track_cost)

    constr = []
    constr.append(r[:n] == x0)

    prob = cp.Problem(obj, constr)
    prob.solve()

    z0 = np.append(x0 - r[:n].value, r.value)

    plt.figure()
    plt.plot(z0[n::2], z0[n+1::2], label="layered ref")
    plt.close()

    return z0


def reference_tracking(A, B, Q, R, q, N, F, r, n, m, x0, lqr):
    # Riccati recursions
    P, p, c, K, M, v = riccati_recursion(A, B, Q, q, R, N)

    # Defining augmented state as [err, ref]
    print("reference", r)
    z0 = np.append(x0 - r[:, 0], r.ravel(order='F'))

    # Compute the trajectory
    u_lqr = np.zeros([m, N])
    x_lqr = np.zeros([n, N+1])
    z_lqr = np.zeros([n * (N+2), N+1])

    x_lqr[:, 0] = x0

    for i in range(N+1):
        z_lqr[:n, i] = x_lqr[:, i] - r[:, i]
        z_lqr[n:(N-i+n)*n, i] = r[:, i:].ravel(order='F')


    for i in range(N):
        u_lqr[:, i] = -K[N-i-1] @ z_lqr[:, i] - v[N-i-1]
        z_lqr[:, i+1] = A @ z_lqr[:, i] + B @ u_lqr[:, i]

    x_lqr = z_lqr[:n, :] + r

    plt.figure()
    #plt.plot(z_lqr[0, :], z_lqr[1, :], label="ref tracking")
    plt.plot(z_lqr[0, :]+r[0, :], label="admm")
    #plt.plot(r[0, :], r[1, :], label="ref")
    plt.plot(r[0, :], 'r--', label="reference")
    plt.xlabel("time")
    plt.ylabel("state")
    plt.legend()
    plt.show()

    return P, p, c, x_lqr, u_lqr


def main():
    N = 12  # horizon

    n = 2  # state dim
    m = 2  # input dim

    # n-dim chain of integrators
    A = np.diag(np.ones(n)) + np.diag(np.ones(n-1),1)
    print(A)

    # actuators enter through the bottom of the chain
    B = np.zeros((n,m))
    B[-m:,:] = np.eye(m)
    print(B)

    # Cost weight matrix for lqr
    Q = np.array([[10, 0], [0, 1]])

    Q_list = [Q for i in range(N+1)]
    R = 0.001 * np.eye(m)
    q = np.ones(2)
    q_list = [q for i in range(N+1)]
    q_list.append(0)

    x0 = np.array([0, 0])  # initial state

    # Define matrix F s.t. e = Fz
    F = np.hstack([np.eye(n), np.zeros([n, n * (N + 1)])])

    # TODO: How to set the optimal mu value?
    mu = np.zeros((n, N + 1))

    rho = 50

    # Matrices for z-dimensions
    E1 = np.vstack([np.eye(n), np.zeros([n * N, n])])
    E2 = np.vstack([np.zeros([n, n]), np.eye(n), np.zeros([n * (N - 1), n])])
    Z = np.eye(n * (N + 1), n * (N + 1), 1)

    # Initialize the cost weight matrix
    Q_bar = np.block(
        [[rho / 2 * Q, np.zeros([n, (N + 1) * n])], [np.zeros([(N + 1) * n, n]), np.zeros([(N + 1) * n, (N + 1) * n])]])

    # Dynamics for z
    A_bar = np.block([[A, E2.T - A @ E1.T], [np.zeros([n * (N + 1), n]), Z]])
    B_bar = np.vstack([B, np.zeros([n * (N + 1), m])])

    # According to layering notes, q = F.T @ mu
    q_list = []
    for i in range(N + 1):
        q_list.append(rho * F.T @ mu[:, i])
    q_list.append(0)

    # Defining the same cost Q_bar for all time steps
    Q_bar_list = []
    Q_bar_list = [Q_bar for i in range(N + 1)]
    print(is_pos_def(Q_bar_list[N]))

    # Riccati recursions
    P, p, c, K, M, v = riccati_recursion(A_bar, B_bar, Q_bar_list, q_list, R, N)

    # Solve for the reference trajectory using the cost
    z0 = layered_planning(x0, np.zeros((n, N + 1)), np.zeros((n, N + 1)), n, N, rho)
    r = np.reshape(z0[n:], (n, N + 1), order='F')

    # Solve for the reference tracking problem
    P, p, c, x, u = reference_tracking(A_bar, B_bar, Q_bar_list, R, q_list, N, F, r, n, m, x0, False)

    # Run this iteratively with ADMM
    vk = np.zeros((n, N + 1))
    rhok = cp.Parameter(nonneg=True)
    rhok.value = rho

    xk = np.zeros((n, N + 1))
    xk = x
    uk = np.zeros((m, N))

    err = 100

    while err >= 1:

        # Solve the planning problem
        # import pdb; pdb.set_trace()
        P, p, c, K, M, v = riccati_recursion(A_bar, B_bar, Q_bar_list, q_list, R, N)
        z0_k = layered_planning(x0, xk, vk, n, N, rhok.value)

        # Planned reference reshaped
        rk = np.reshape(z0_k[n:], (n, N + 1), order='F')
        print("ref rk", rk)

        # Solve the tracking problem
        P, p, c, x, u = reference_tracking(A_bar, B_bar, Q_bar_list, R, q_list, N, F, rk, n, m, x0, False)

        sxk = rhok.value * (xk - x).flatten()
        suk = rhok.value * (uk - u).flatten()

        dual_res_norm = np.linalg.norm(np.hstack([sxk, suk]))
        p_res_norm = np.linalg.norm(rk - xk)

        # update rhok and rescale vk
        if p_res_norm > 10 * dual_res_norm:
            rhok.value = 2 * rhok.value
            vk = vk / 2
        elif dual_res_norm > 10 * p_res_norm:
            rhok.value = rhok.value / 2
            vk = vk * 2

        # update v
        xk = x
        uk = u
        vk = vk + xk - rk
        print("Vk", vk)

        Q_bar_list = []
        Q_bar_list = [(rhok.value / 2) * Q_bar for i in range(N + 1)]
        # Q_bar_list = [Q_bar for i in range(N+1)]
        print(is_pos_def(Q_bar_list[N]))

        # According to layering notes, q = F.T @ mu
        q_list = []
        for i in range(N + 1):
            q_list.append(rhok.value * F.T @ vk[:, i])
        q_list.append(0)

        residual = rk - xk
        err = np.trace(residual.T @ residual)
        print("Residual", err)


if __name__ == '__main__':
    main()



