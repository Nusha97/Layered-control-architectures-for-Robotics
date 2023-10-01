import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from trajax import optimizers
import matplotlib.pyplot as plt
import cvxpy as cp
import pickle


horizon = 40
dt = 0.1
eq_point = jnp.array([0, jnp.pi, 0, 0])


@jax.jit
def cartpole(state, action, timestep, params=(10.0, 1.0, 0.5)):
  """Classic cartpole system.

  Args:
    state: state, (4, ) array
    action: control, (1, ) array
    timestep: scalar time
    params: tuple of (MASS_CART, MASS_POLE, LENGTH_POLE)

  Returns:
    xdot: state time derivative, (4, )
  """
  del timestep  # Unused

  mc, mp, l = params
  g = 9.81

  q = state[0:2]
  qd = state[2:]
  s = jnp.sin(q[1])
  c = jnp.cos(q[1])

  H = jnp.array([[mc + mp, mp * l * c], [mp * l * c, mp * l * l]])
  C = jnp.array([[0.0, -mp * qd[1] * l * s], [0.0, 0.0]])

  G = jnp.array([[0.0], [mp * g * l * s]])
  B = jnp.array([[1.0], [0.0]])

  CqdG = jnp.dot(C, jnp.expand_dims(qd, 1)) + G
  f = jnp.concatenate(
      (qd, jnp.squeeze(-jsp.linalg.solve(H, CqdG, assume_a='pos'))))

  v = jnp.squeeze(jsp.linalg.solve(H, B, assume_a='pos'))
  g = jnp.concatenate((jnp.zeros(2), v))
  xdot = f + g * action

  return xdot


def angle_wrap(th):
  return (th) % (2 * jnp.pi)


def state_wrap(s):
  return jnp.array([s[0], angle_wrap(s[1]), s[2], s[3]])


def squish(u):
  return 5 * jnp.tanh(u)


def ilqr_cost(x, u, t):
  err = state_wrap(x - eq_point)
  stage_cost = 0.1 * jnp.dot(err, err) + 0.1 * jnp.dot(u, u)
  final_cost = 10 * jnp.dot(err, err)
  return jnp.where(t == horizon, final_cost, stage_cost)


def dynamics(x, u, t):
  return x + dt * cartpole(x, squish(u), t)


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

    return int(iter_i), float((X[-1] - eq_point).T @ (X[-1] - eq_point))


def admm(x0):
    rho = 25
    T = horizon + 1
    n = 4
    m = 1

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
    residual = []
    il_it = 0
    while err >= 1e-1:
        k += 1
        # update r
        r_suprob.solve()

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


    return il_it, err, residual


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
    admm_res = []
    np.random.seed(10)
    rng = np.random.default_rng()

    for i in range(num_iter):
        x0 = rng.uniform(0, 1, 4)
        # x0[0] = rng.uniform(0, 1)
        ilqr_iter.append(ilqr(x0))
        a, b, c = admm(x0)
        admm_iter.append([a, b])
        admm_res.append(c)
        # admm_iter.append(admm(x0))
        print(ilqr_iter[i])
        print(admm_iter[i])

    admm_iter = np.vstack(admm_iter)
    ilqr_iter = np.vstack(ilqr_iter)

    print(ilqr_iter)
    print(admm_iter)



    print("Mean ilqr", np.mean(ilqr_iter[:, 0]))
    print("Std ilqr", np.std(ilqr_iter[:, 0]))
    print("Mean admm", np.mean(admm_iter[:, 0]))
    print("Std admm", np.std(admm_iter[:, 0]))
    print("ILQR convergence rate", np.sum(np.where(ilqr_iter[:, 1] <= 1, 1, 0))/num_iter)
    print("ADMM convergence rate", np.sum(np.where(admm_iter[:, 1] <= 1, 1, 0))/num_iter)

    save_object(ilqr_iter, "./cp_ilqr.pkl")
    save_object(admm_iter, "./cp_admm.pkl")
    save_object(admm_res, "./cp_admm_res.pkl")



if __name__ == '__main__':
    main()