import numpy as np
import scipy.special as sps
import cvxpy as cp

def diff_coeff(n, t, dx_order):
    assert dx_order <= n
    count = n - dx_order # Number of nonzero elements
    t_powers = t ** np.arange(count)
    multln = sps.gammaln(np.arange(count) + dx_order + 1) - sps.gammaln(np.arange(count)+1)
    mult = np.exp(multln)
    return np.concatenate([np.zeros(dx_order), t_powers * mult])

def jerk_coeff(n, t1, t2):
    mult = (np.arange(n-3)+3) * (np.arange(n-3)+2)
    t1_powers = t1 ** (np.arange(n-3)+1)
    t2_powers = t2 ** (np.arange(n-3)+1)
    return np.concatenate([np.zeros(3), mult*(t2_powers - t1_powers)])

def min_jerk(waypoints, t, n, num_steps, P=None, rho=1):
    ''' Generate min jerk trajectory with regularization P
    '''

    #TODO: make this work for multi-dimensional trajectory!!!

    # Construct the optimization problem
    coeff = cp.Variable((len(waypoints)-1, n))
    ref = cp.Variable(num_steps + 1)
    objective = 0
    constr = []
    # First waypoint
    constr += [
        diff_coeff(n, t[0], 0) @ coeff[0] == waypoints[0],
        diff_coeff(n, t[0], 1) @ coeff[0] == 0,
        diff_coeff(n, t[0], 2) @ coeff[0] == 0
    ]
    objective += cp.sum_squares(jerk_coeff(n, t[0], t[1]) @ coeff[0])
    # Intermediate waypoints
    for i in range(1, len(waypoints)-1):
        constr += [
            diff_coeff(n, t[i], 0) @ coeff[i-1] == waypoints[i],
            diff_coeff(n, t[i], 0) @ coeff[i] == waypoints[i],
            diff_coeff(n, t[i], 1) @ coeff[i] == diff_coeff(n, t[i], 1) @ coeff[i-1],
            diff_coeff(n, t[i], 2) @ coeff[i] == diff_coeff(n, t[i], 2) @ coeff[i-1]
        ]
        objective += cp.sum_squares(jerk_coeff(n, t[0], t[1]) @ coeff[i])
    # Last waypoint
    constr += [
        diff_coeff(n, t[-1], 0) @ coeff[-1] == waypoints[-1],
        diff_coeff(n, t[-1], 1) @ coeff[-1] == 0,
        diff_coeff(n, t[-1], 2) @ coeff[-1] == 0
    ]
    # Construct trajectory
    k = 0
    for i in range(num_steps + 1):
        if i / num_steps > t[k+1]:
            k = k + 1
        constr += [ref[i] == diff_coeff(n, i/num_steps, 0) @ coeff[k]]
    # Add regularization
    if P is not None:
        #TODO: something that needs to be fixed for higher-dimensional sys
        P12 = P[0,1:]
        P22 = P[1:,1:]
        penalty = cp.quad_form(ref, P22) + 2*waypoints[0] * P12@ref +\
                P[0,0] * waypoints[0] ** 2
        objective = objective + rho * penalty
    # Solve
    prob = cp.Problem(cp.Minimize(objective), constr)
    prob.solve(verbose=False)
    if prob.status != cp.OPTIMAL:
        print('Failed to generate trajectory')
        return None
    # Construct return value
    if P is not None:
        tracking_cost = penalty.value
    else:
        tracking_cost = None
    return coeff.value, ref.value, tracking_cost
