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

def _facln(n, i):
    ''' Helper function that computes ln( n*(n-i)*...*(n-i+1) ) '''
    return np.log(n - np.arange(i)).sum()

def cost_matrix(n, k, T):
    ''' Return the quadratic cost matrix corresponds to the cost function
            \int_0^T (\frac{\partial^k x}{\partial t^k})^2 dt
    Input:
        - n: degree of the polynomial
        - k: order of derivative that we integrate over
    Return:
        - H: cost matrix
    '''
    H = np.zeros((n+1, n+1))
    for i in range(n-k+1):
        for j in range(n-k+1):
            power = 2*n-2*k-i-j+1
            Hij_ln = _facln(n-i, k) + _facln(n-j, k) - np.log(power)
            H[i,j] = np.exp(Hij_ln + np.log(T) * power)
    return H

def continuity_constr(n, order, coeff1, coeff2, waypoint, T1):
    ''' Return a list of continuity constraints enforced at p_1(T1) and p_2(0).
    In addition, enforce coeff
    Input:
        - n:        Integer, order of the polynomial
        - order:    Integer, order of continuity enforced
        - coeff1:   cp.Variable, coefficients of polynomial p_1
        - coeff2:   cp.Variable, coefficients of polynomial p_2
    Return:
        - constr:   list of cp.Constraint
    '''
    pass

def boundary_cond(n, coeff, bcs, T):
    ''' Return a list of bc constraints enforced at p(T)
    Input:
        - n:        Integer, order of polynomial
        - coeff:    coefficients of the polynomial p(t)
        - bcs:      list of boundary condition values, ordered as
                    [p(0), p'(0), p''(0), ...]
    '''
    pass

def min_jerk(waypoints, t, n, num_steps, P=None, rho=1, threshold=10):
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
        #constr += [objective <= threshold]
    # Solve
    prob = cp.Problem(cp.Minimize(objective), constr)
    #prob = cp.Problem(cp.Minimize(penalty), constr)
    prob.solve(verbose=False)
    print(coeff.value)
    #print('threshold on jerk is {:6.3f},\t jerk is {:6.3f},\t penalty is {:2.3f}'.format(threshold, objective.value, penalty.value))
    if prob.status != cp.OPTIMAL:
        print('Failed to generate trajectory')
        return None
    print('jerk: {:6.3f},\t penalty:{:6.3f}'.format(objective.value-rho*penalty.value, penalty.value))
    # Construct return value
    if P is not None:
        tracking_cost = penalty.value
    else:
        tracking_cost = None
    return coeff.value, ref.value, tracking_cost
