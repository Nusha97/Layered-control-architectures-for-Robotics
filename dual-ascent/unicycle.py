import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
import jax.scipy as jsp
# import warnings
from trajax import optimizers
import matplotlib.pyplot as plt
import pickle


horizon = 20
dt = 0.1
eq_point = jnp.array([3, 2, 0])


@jax.jit
def car(x, u, t):
      del t
      return jnp.array([u[0] * jnp.cos(x[2]), u[0] * jnp.sin(x[2]), u[1]])


def plot_car(x, param, col='black', col_alpha=1):
    w = param.l / 2
    x_rl = x[:2] + 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
    x_rr = x[:2] - 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
    x_fl = x_rl + param.l * np.array([np.cos(x[2]), np.sin(x[2])])
    x_fr = x_rr + param.l * np.array([np.cos(x[2]), np.sin(x[2])])
    x_plot = np.concatenate((x_rl, x_rr, x_fr, x_fl, x_rl))
    plt.plot(x_plot[0::2], x_plot[1::2], linewidth=2, c=col, alpha=col_alpha)
    plt.scatter(x[0], x[1], marker='.', s=200, c=col, alpha=col_alpha)


def angle_wrap(theta):
    """
    Function to wrap angles greater than pi
    :param theta: heading angle of system
    :return: wrapped angle
    """
    return (theta) % (2 * np.pi)


def state_wrap(s):
  return jnp.array([s[0], s[1], angle_wrap(s[2])])


def dynamics(x, u, t):
  # return rk4(car, dt=dt)
  return x + dt * car(x, u, t)


def rollout(x0, U, dynamics, T):
  n = len(x0)
  x = np.zeros((n,T))
  x[:, 0] = x0
  for t in range(T-1):
    x[:, t+1] = dynamics(x[:, t], U[:, t], t)
  return x


def ilqr(x0):
    def cost(x, u, t):
        err = state_wrap(x - eq_point)
        stage_cost = 0.1 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
        final_cost = 1000 * jnp.dot(err, err)
        return jnp.where(t == horizon, final_cost, stage_cost)

    X, U, _, _, _, _, iter_i = optimizers.ilqr(
        cost,
        dynamics,
        x0,
        jnp.zeros((horizon, 1)),
        maxiter=200
    )

    X = X.T
    param = lambda: None  # Lazy way to define an empty class in python
    param.nbData = horizon + 1
    param.l = .25  # Length of the car
    param.Mu = np.array([eq_point[0], eq_point[1], 0])  # Viapoint (x1,x2,theta,phi)

    # Plotting
    # ===============================
    plt.figure()
    # plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.plot(X[0, :], X[1, :], c='black')

    nb_plots = 15
    for i in range(nb_plots):
        plot_car(X[:, int(i * param.nbData / nb_plots)], param, 'black', 0.1 + 0.9 * i / nb_plots)
    plot_car(X[:, -1], param, 'black')

    plot_car(param.Mu, param, 'red')
    plt.scatter(param.Mu[0], param.Mu[1], color='r', marker='.', s=200, label="Desired pose")
    plt.ylim(-0.1, 3)
    plt.xlim(-0.1, 3.5)
    plt.legend()

    # plt.show()
    return int(iter_i), float((X.T[-1] - eq_point).T @ (X.T[-1] - eq_point))


def admm(x0):
    rho = 25
    T = horizon + 1
    n = 3
    m = 2

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

    # this is probably a source of error because I can't implement angle
    # wrapping using cvxpy
    # stage_err = cp.hstack([(r[:, t] - eq_point) for t in range(T-1)])
    stage_err = cp.hstack([(r[:, t]) for t in range(T - 1)])
    final_err = r[:, -1] - eq_point

    stage_cost = 0.1 * cp.sum_squares(stage_err)
    final_cost = 1000 * cp.sum_squares(final_err)
    utility_cost = stage_cost + final_cost
    admm_cost = rhok * cp.sum_squares(r - xk + vk)

    # Obstacle are convex polytopes
    # Define the vertices of the polytope
    vertices = np.array([[1, 1, 0], [1, 2, 0], [2, 1, 0]])

    # Create a matrix of coefficients for the linear inequalities
    a1 = np.array([1, 0, 0])
    a2 = np.array([1, 0, 0])
    a3 = np.array([0, 1, 0])
    a4 = np.array([0, 1, 0])
    # Ac = np.vstack([a1, a2, a3])

    # Define the right-hand side of the inequalities (the bounds)
    # xc = np.array([[1.5, 1.5, 0], [1, 1, 0], [1, 1, 0]])
    b = np.array([1, -0.5, 3])
    constr = []
    # constr.append(r[:, 0] == xk[:, 0])
    # for t in range(T):
    # constr.append(a1.T @ (r[:, t] - xc[0, :]))
    # constr.append(a2.T @ (r[:, t] - xc[1, :]))
    # constr.append(a3.T @ (r[:, t] - xc[2, :]))
    # constr.append(Ac @ r[:, t] >= b)
    # constr.append(a1.T @ r[:, t] <= b[0])
    # constr.append(a2.T @ r[:, t] >= b[1])
    # constr.append(r[0, :-int(T/2)] <= 1)
    # constr.append(r[0, :-int(T/2)] >= 0)
    # constr.append(r[1, int(T/2):] >= 1.5)
    # constr.append(r[1, int(T/2):] <= 2.5)
    r_suprob = cp.Problem(cp.Minimize(utility_cost + admm_cost), constr)

    @jax.jit
    def solve_xu_subproblem(rk, vk, uk, rho):
        def cost(x, u, t):
            err = rk[:, t] - x + vk[:, t]
            # err = state_wrap(err)
            # trying without state_wrap to be consistent with planning problem
            stage_cost = rho / 2 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
            final_cost = rho / 2 * jnp.dot(err, err)
            return jnp.where(t == horizon, final_cost, stage_cost)

        X, U, _, _, _, _, _ = optimizers.ilqr(
            cost,
            dynamics,
            x0,
            uk.T,
            maxiter=10
        )
        return X, U

    # run ADMM algorithms
    # K = 50
    rk = cp.Parameter((n, T))
    rk.value = np.zeros((n, T))
    xk.value = np.zeros((n, T))
    uk.value = np.zeros((m, T - 1))
    vk.value = np.zeros((n, T))
    rhok.value = 25

    # for k in np.arange(K):
    k = 0
    err = 100
    residual = []
    while err >= 1e-2:
        k += 1
        # update r
        r_suprob.solve()

        # update x u
        rk.value = r.value
        # print(rk.value)

        x, u = solve_xu_subproblem(jnp.array(rk.value), jnp.array(vk.value), jnp.array(uk.value), jnp.array(rhok.value))

        # compute residuals
        sxk = rhok.value * (xk.value - x.T).flatten()
        suk = rhok.value * (uk.value - u.T).flatten()
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
        xk.value = np.array(x.T)
        uk.value = np.array(u.T)
        vk.value = vk.value + rk.value - xk.value

        err = np.trace((rk.value - xk.value).T @ (rk.value - xk.value))
        residual.append(err)

    param = lambda: None  # Lazy way to define an empty class in python
    param.nbData = T
    param.l = .25  # Length of the car
    param.Mu = np.array([eq_point[0], eq_point[1], 0])  # Viapoint (x1,x2,theta,phi)

    # Plotting
    # ===============================
    plt.figure()
    # plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.plot(xk.value[0, :], xk.value[1, :], c='black')

    nb_plots = 15
    for i in range(nb_plots):
        plot_car(xk.value[:, int(i * param.nbData / nb_plots)], param, 'black', 0.1 + 0.9 * i / nb_plots)
    plot_car(xk.value[:, -1], param, 'black')

    plot_car(param.Mu, param, 'red')
    plt.scatter(param.Mu[0], param.Mu[1], color='r', marker='.', s=200, label="Desired pose")
    # plt.plot(np.array([1, 1]), np.array([1, 2]), color='b', linewidth=4)
    # plt.plot(np.array([2, 1]), np.array([1, 2]), color='b', linewidth=4)
    # plt.plot(np.array([2, 1]), np.array([1, 1]), color='b', linewidth=4)
    # plt.plot(np.array([1, 1]), np.array([0, 1.5]), color='b', linewidth=4)
    # plt.plot(np.array([1, 4]), np.array([1.5, 1.5]), color='b', linewidth=4)
    # plt.plot(np.array([0, 0]), np.array([0, 2.5]), color='b', linewidth=4)
    # plt.plot(np.array([0, 4]), np.array([2.5, 2.5]), color='b', linewidth=4)
    plt.ylim(-0.1, 3)
    plt.xlim(-0.1, 3.5)
    plt.legend()

    # plt.show()

    return k * (1 + 10), err, residual


def admm_corridor(x0):
    rho = 25
    T = horizon + 1
    n = 3
    m = 2

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

    # this is probably a source of error because I can't implement angle
    # wrapping using cvxpy
    # stage_err = cp.hstack([(r[:, t] - eq_point) for t in range(T-1)])
    stage_err = cp.hstack([(r[:, t]) for t in range(T - 1)])
    final_err = r[:, -1] - eq_point

    stage_cost = 0.1 * cp.sum_squares(stage_err)
    final_cost = 1000 * cp.sum_squares(final_err)
    utility_cost = stage_cost + final_cost
    admm_cost = rhok * cp.sum_squares(r - xk + vk)

    # Obstacle are convex polytopes
    # Define the vertices of the polytope
    vertices = np.array([[1, 1, 0], [1, 2, 0], [2, 1, 0]])

    # Create a matrix of coefficients for the linear inequalities
    a1 = np.array([1, 0, 0])
    a2 = np.array([1, 0, 0])
    a3 = np.array([0, 1, 0])
    a4 = np.array([0, 1, 0])
    # Ac = np.vstack([a1, a2, a3])

    # Define the right-hand side of the inequalities (the bounds)
    # xc = np.array([[1.5, 1.5, 0], [1, 1, 0], [1, 1, 0]])
    b = np.array([1, -0.5, 3])
    constr = []
    # constr.append(r[:, 0] == xk[:, 0])
    # for t in range(T):
    # constr.append(a1.T @ (r[:, t] - xc[0, :]))
    # constr.append(a2.T @ (r[:, t] - xc[1, :]))
    # constr.append(a3.T @ (r[:, t] - xc[2, :]))
    # constr.append(Ac @ r[:, t] >= b)
    # constr.append(a1.T @ r[:, t] <= b[0])
    # constr.append(a2.T @ r[:, t] >= b[1])
    constr.append(r[1, int(T / 2):] >= 1.5)
    constr.append(r[1, int(T / 2):] <= 2.5)
    r_subprob = cp.Problem(cp.Minimize(utility_cost + admm_cost), constr)
    constr.append(r[0, :-int(T / 2)] <= 1)
    constr.append(r[0, :-int(T / 2)] >= 0)
    r_suprob = cp.Problem(cp.Minimize(utility_cost + admm_cost), constr)


    @jax.jit
    def solve_xu_subproblem(rk, vk, uk, rho):
        def cost(x, u, t):
            err = rk[:, t] - x + vk[:, t]
            # err = state_wrap(err)
            # trying without state_wrap to be consistent with planning problem
            stage_cost = rho / 2 * jnp.dot(err, err) + 0.01 * jnp.dot(u, u)
            final_cost = rho / 2 * jnp.dot(err, err)
            return jnp.where(t == horizon, final_cost, stage_cost)

        X, U, _, _, _, _, it = optimizers.ilqr(
            cost,
            dynamics,
            x0,
            uk.T,
            maxiter=10
        )
        return X, U, it

    # run ADMM algorithms
    # K = 50
    rk = cp.Parameter((n, T))
    rk.value = np.zeros((n, T))
    xk.value = np.zeros((n, T))
    uk.value = np.zeros((m, T - 1))
    vk.value = np.zeros((n, T))
    rhok.value = 25

    # for k in np.arange(K):
    k = 0
    err = 100
    status = True
    residual = []
    il_it = 0
    while err >= 1e-2:
        k += 1
        # update r
        try:
            r_suprob.solve()
            status = True
        except:
            r_subprob.solve()
            status = False

        # update x u
        rk.value = r.value
        # print(rk.value)

        x, u, it = solve_xu_subproblem(jnp.array(rk.value), jnp.array(vk.value), jnp.array(uk.value), jnp.array(rhok.value))

        # compute residuals
        sxk = rhok.value * (xk.value - x.T).flatten()
        suk = rhok.value * (uk.value - u.T).flatten()
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
        xk.value = np.array(x.T)
        uk.value = np.array(u.T)
        vk.value = vk.value + rk.value - xk.value

        err = np.trace((rk.value - xk.value).T @ (rk.value - xk.value))
        residual.append(err)
        il_it += int(it) + 1

    param = lambda: None  # Lazy way to define an empty class in python
    param.nbData = T
    param.l = .25  # Length of the car
    param.Mu = np.array([eq_point[0], eq_point[1], 0])  # Viapoint (x1,x2,theta,phi)

    # Plotting
    # ===============================
    plt.figure()
    # plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.plot(xk.value[0, :], xk.value[1, :], c='black')

    nb_plots = 15
    for i in range(nb_plots):
        plot_car(xk.value[:, int(i * param.nbData / nb_plots)], param, 'black', 0.1 + 0.9 * i / nb_plots)
    plot_car(xk.value[:, -1], param, 'black')

    plot_car(param.Mu, param, 'red')
    plt.scatter(param.Mu[0], param.Mu[1], color='r', marker='.', s=200, label="Desired pose")
    # plt.plot(np.array([1, 1]), np.array([1, 2]), color='b', linewidth=4)
    # plt.plot(np.array([2, 1]), np.array([1, 2]), color='b', linewidth=4)
    # plt.plot(np.array([2, 1]), np.array([1, 1]), color='b', linewidth=4)
    # plt.plot(np.array([1, 1]), np.array([0, 1.5]), color='b', linewidth=4)
    # plt.plot(np.array([1, 4]), np.array([1.5, 1.5]), color='b', linewidth=4)
    # plt.plot(np.array([0, 0]), np.array([0, 2.5]), color='b', linewidth=4)
    # plt.plot(np.array([0, 4]), np.array([2.5, 2.5]), color='b', linewidth=4)
    plt.ylim(-0.1, 3)
    plt.xlim(-0.1, 3.5)
    plt.legend()

    # plt.show()

    return il_it, err, status, residual


def save_object(obj, filename):
    """
    Save object as a pickle file
    :param obj: Data to be saved
    :param filename: file path
    :return: None
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def main():

    # x0 = jnp.array([0.0, 0.2, 0])

    num_iter = 20
    ilqr_iter = []
    admm_iter = []
    admm_corridor_iter = []
    ilqr_res = []
    admm_res = []
    admm_corridor_res = []
    np.random.seed(0)
    rng = np.random.default_rng()


    for i in range(num_iter):
        x0 = rng.standard_normal(3)
        x0[0] = rng.uniform(0, 1)
        ilqr_iter.append(ilqr(x0))
        a, b, c = admm(x0)
        admm_iter.append([a, b])
        admm_res.append(c)
        # admm_iter.append(admm(x0))
        a, b, c, d = admm_corridor(x0)
        admm_corridor_iter.append([a, b, c])
        admm_corridor_res.append(d)
        # admm_corridor_iter.append(admm_corridor(x0))
        print(ilqr_iter[i])
        print(admm_iter[i])
        print(admm_corridor_iter[i])

    admm_iter = np.vstack(admm_iter)
    print(admm_iter.shape)
    ilqr_iter = np.vstack(ilqr_iter)
    admm_corridor_iter = np.vstack(admm_corridor_iter)

    print(ilqr_iter)
    print(admm_iter)

    print("Mean ilqr", np.mean(ilqr_iter[:, 0]))
    print("Std ilqr", np.std(ilqr_iter[:, 0]))
    print("Mean admm", np.mean(admm_iter[:, 0]))
    print("Std admm", np.std(admm_iter[:, 0]))
    print("Mean corridor", np.mean(admm_corridor_iter[:, 0]))
    print("Std corridor", np.std(admm_corridor_iter[:, 0]))
    print("ILQR convergence rate", sum(np.where(ilqr_iter[:, 1] <= 1, 1, 0)) / num_iter)
    print("ADMM convergence rate", sum(np.where(admm_iter[:, 1] <= 1, 1, 0)) / num_iter)
    print("ADMM Corridor convergence rate", sum(np.where(admm_corridor_iter[:, 1] <= 1, 1, 0)) / num_iter)
    # print(ilqr_iter)
    # print(admm_iter)
    # print(admm_corridor_iter)
    #
    # print("Mean ilqr", np.mean(ilqr_iter[0]))
    # print("Std ilqr", np.std(ilqr_iter[0]))
    # print("Mean admm", np.mean(admm_iter[0]))
    # print("Std admm", np.std(admm_iter[0]))
    # print("Mean corridor", np.mean(admm_corridor_iter[0]))
    # print("Std corridor", np.std(admm_corridor_iter[0]))

    # plt.figure()
    # admm_err = np.mean(np.vstack(admm_res), axis=0)
    # admm_std = np.std(np.vstack(admm_res), axis=0)
    #
    # plt.errorbar(np.arange(num_iter), admm_err, admm_std)

    save_object(ilqr_iter, "./uni_ilqr.pkl")
    save_object(admm_iter, "./uni_admm.pkl")
    save_object(admm_corridor_iter, "./uni_admm_corridor.pkl")
    save_object(admm_res, "./uni_admm_res.pkl")
    save_object(admm_corridor_res, "./uni_admm_corridor_res.pkl")



if __name__ == '__main__':
    main()


