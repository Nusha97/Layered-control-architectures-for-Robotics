# This file contains the wrapper for quadrotor trajectory generation

import numpy as np
import cvxpy as cp

from .trajutils import _diff_coeff, _facln, _cost_matrix
from .quadratic import _continuity_constr, _boundary_cond, min_jerk_1d

############################################################
# Generation of min-jerk trajectories
############################################################

def _derivative_bounds(coeff, waypoints, ts, n, vmax, amax):
    ''' Generate a list of constraints for velocity and acceleration
    '''
    # Duration for each segment
    ts = np.array(ts)
    durations = ts[1:] - ts[:-1]
    # compute constraints
    poly_order = coeff.shape[1] - 1
    constr = []
    for i in range(1, len(waypoints)-1):
        dx = _diff_coeff(poly_order, 0, 1) @ coeff[i]
        ddx = _diff_coeff(poly_order, 0, 2) @ coeff[i]
        constr += [
                -vmax <= dx, dx <= vmax,
                -amax <= ddx, ddx <= amax
        ]
    return constr

def quad_min_jerk_setup(waypoints, ts, n, num_steps, vmax, amax):
    ''' Sets up the min jerk problem by calling the 1d helper function
    Note that this function generates a reference trajectory of length 4T
    '''
    # Assume we get waypoints for x, y, z, yaw
    p = waypoints.shape[1]
    assert(p == 4)

    # Generate the variables and constraints for each dimension
    objective, constrs, refs, coeffs = 0, [], [], []
    for i in range(p):
        o, c, r, co = min_jerk_1d(waypoints[:,i], ts, n, num_steps)
        objective += o
        constrs += c
        refs.append(r)
        # Add in the velocity and acceleration constraints
        constrs += _derivative_bounds(co, waypoints[:,i], ts, n, vmax, amax)
        coeffs.append(co)
    # Stitch them into global reference trajectory
    ref = cp.vstack(refs).flatten()
    return objective, constrs, ref, coeffs


def generate(waypoints, ts, n, num_steps, p, vmax=2, amax=2):
    ''' Wrapper for generating trajectory. For now only takes quadratic
    regularization.
    Return:
        ref:        reference trajectory
    '''
    objective, constr, ref, coeff = quad_min_jerk_setup(waypoints, ts, n,
            num_steps, vmax, amax)
    prob = cp.Problem(cp.Minimize(objective), constr)
    prob.solve()
    if prob.status != cp.OPTIMAL:
        return None
    else:
        return ref.value, np.array([c.value for c in coeff])
