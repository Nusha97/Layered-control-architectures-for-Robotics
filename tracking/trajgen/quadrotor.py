import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg as spl

from .trajutils import _diff_coeff, _facln, _cost_matrix
from .nonlinear import _coeff_constr_A, _coeff_constr_b, projectcoeff

class MinJerkReg(nn.Module):
    ''' A class that takes in a trajectory and computes its cost
    '''
    def __init__(self, ts, order, regularizer, coeff):
        super().__init__()
        # Compute cost matrices (note its shared across dimensions)
        num_seg = len(ts-1)
        durations = ts[1:] - ts[:-1]
        cost_mat = spl.block_diag(*[_cost_matrix(order, 3, d) for d in durations])
        self.ts = ts
        self.cost_mat = torch.tensor(cost_mat)
        self.regularizer = regularizer
        self.coeff = nn.Parameter(coeff)

    def forward(self, x0, p, rho, num_steps):
        # Compute jerk
        cost = 0
        for pp in range(p):
            cost += torch.dot(self.coeff[pp].reshape(-1),
                              self.cost_mat @ self.coeff[pp].reshape(-1))
            #print('Jerk penalty is {:10.3f}'.format(cost))
        # Compute regularizer
        if self.regularizer is not None:
            ref = coeff2traj(self.coeff, self.ts, num_steps)[1].T.flatten()
            reg = self.regularizer.pred(x0, ref)[0]
            cost += rho *  self.regularizer.pred(x0, ref)[0]
        #print('Regularizer is {:10.3f} x {:6.3f}'.format(reg, rho))
        #print('-'*20)
        return cost

def coeff2traj(coeffs, ts, numsteps, fullstate=True):
    ''' Constructs a trajectory from polynomial coefficients
        Construct the 14-D state from the 4 sets of coefficients
    '''
    if fullstate == True:
        p = 14
    else:
        p = coeffs.shape[0]
    ref = torch.zeros(p, numsteps)
    times = torch.linspace(ts[0], ts[-1], numsteps)
    k = 0
    for i, tt in enumerate(times):
        if tt > ts[k+1]: k += 1
        # *i*-th timestep, uses *k*-th set of coefficients
        ref[:3, i] = torch.tensor(
                _diff_coeff(coeffs.shape[2]-1, tt-ts[k], 0)) @ coeffs[:3,k,:].T
        if fullstate:
            ref[3:6, i] = torch.tensor(
                    _diff_coeff(coeffs.shape[2]-1, tt-ts[k], 1)) @ coeffs[:3,k,:].T
            ref[6:9, i] = torch.tensor(
                    _diff_coeff(coeffs.shape[2]-1, tt-ts[k], 2)) @ coeffs[:3,k,:].T
            ref[9:12, i] = torch.tensor(
                    _diff_coeff(coeffs.shape[2]-1, tt-ts[k], 3)) @ coeffs[:3,k,:].T
            ref[12, i] = torch.tensor(
                    _diff_coeff(coeffs.shape[2]-1, tt-ts[k], 0)) @ coeffs[3,k,:].T
            ref[13, i] = torch.tensor(
                    _diff_coeff(coeffs.shape[2]-1, tt-ts[k], 1)) @ coeffs[3,k,:].T
    return times, ref

def generate(waypoints, ts, order, num_steps, p, rho, value_func, coeff0,
        num_iter=30, lr=1e-3):
    ''' Generate trajectory that minimizes the jerk with the dynamic
        regularizer using projected gradient descent.
    Input:
        - coeff0:       np.array(p, #segments, polynomial order), coeffcient
                        for warm-starting
        - num_iter:     Integer, number of GD steps
    '''
    costfn = MinJerkReg(ts, order, value_func, coeff0)
    optimizer = optim.SGD(costfn.parameters(), lr=lr, momentum=0.9)
    x0 = torch.zeros(14)
    x0[:3] = waypoints[:3, 0]
    x0[-2] = waypoints[3, 0]
    for _ in range(num_iter):
        optimizer.zero_grad()
        cost = costfn.forward(x0, p, rho, num_steps)
        cost.backward()
        optimizer.step()
        with torch.no_grad():
            # TODO: figure out a better way to do this so that performance is
            # not compromised
            costfn.coeff.data = projectcoeff(waypoints, ts, costfn.coeff)
    return costfn.coeff.detach()
