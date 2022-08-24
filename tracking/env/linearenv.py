###############################################################################
# Fengjun Yang, 2022
# Linear environment
###############################################################################

import numpy as np
from scipy.linalg import block_diag
import scipy.linalg as spl

from . import baseenv
from . import controller

class LQREnv(baseenv.BaseEnv):
    ''' Linear environment.
    '''
    def __init__(self, A, B, C, Hw, Hv, Q, R, gamma=0.99, sigma_w=0, sigma_v=0):
        '''
        Input:
            - A, B:     np.array, dynamic matrices
            - C:        np.array, linear map from state to observation
            - H_*:      np.array, maps from noise to state / observation
            - Q, R:     np.array, cost matrices
            - sigma_w:  Float, standard deviation of process noise
            - sigma_v:  Float, standard deviation of observation noise
        '''
        self.p, self.q = B.shape
        self.r = C.shape[0]
        self.A, self.B, self.C, self.Hw, self.Hv = A, B, C, Hw, Hv
        self.Q, self.R = Q, R
        self.gamma = gamma
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v

    def step(self, x, u, nonoise=False):
        ''' Step states forward in a batch
        Input:
            - x:        np.array(n, p), batch of states
            - u:        np.array(n, q), batch of inputs
        Return:
            - x_:       np.array(n, p), batch of next states
        '''
        n = x.shape[0]
        if nonoise:
            noise = 0
        else:
            noise = (self.sigma_w * self.Hw @ np.random.randn(n, self.p).T).T
        return (self.A @ x.T + self.B @ u.T).T + noise

    def observe(self, x, nonoise=False):
        ''' Generate an observation from state '''
        n = x.shape[0]
        if nonoise:
            noise = 0
        else:
            noise = (self.sigma_v * self.Hv @ np.random.randn(self.r, n)).T
        return (self.C @ x.T).T + noise

    def reset(self):
        # Nothing to reset here
        pass

    def totracking(self, T):
        ''' Augment the environment for tracking
        Return:
            - envt:     LQREnv, the augmented tracking system
        '''
        # TODO: augment state dimension might be different now that we have an
        # observation model. Need to think more carefully about this.
        Z = np.eye(self.p*T, k=self.p)
        zero = np.zeros((self.p, self.p*T))
        At = np.block([[self.A, zero],[zero.T, Z]])
        #At = block_diag(self.A, Z)
        Bt = np.vstack([self.B, np.zeros((self.p*T, self.q))])
        Ct = block_diag(self.C, np.eye(self.p*T))
        Hwt = block_diag(self.Hw, np.zeros((self.p*T, self.p*T)))
        Hvt = block_diag(self.Hv, np.zeros((self.p*T, self.p*T)))
        E = np.hstack([np.eye(self.p), -np.eye(self.p), \
                np.zeros((self.p, self.p*(T-1)))])
        Qt = E.T @ self.Q @ E
        envt = LQREnv(At, Bt, Ct, Hwt, Hvt, Qt, self.R, sigma_w=self.sigma_w,
                sigma_v=self.sigma_v)
        return envt

    def sampletraj(self, ctrl, T, num_traj, x0s=None, nonoise=False):
        '''
        Input:
            - ctrl:     env.BaseController
        '''
        # Generate x0 if x0 is None
        if x0s is None:
            x0s = np.random.randn(num_traj, self.p)
        else:
            assert num_traj == x0s.shape[0]
        n = num_traj

        # Initialize variables
        p, q = self.B.shape
        xtraj = np.zeros((T, n, self.p))
        xtraj_ = np.zeros((T, n, self.p))
        ytraj = np.zeros((T, n, self.r))
        ytraj_ = np.zeros((T, n, self.r))
        utraj = np.zeros((T, n, self.q))
        rtraj = np.zeros((T, n, 1))

        # Generate trajectory
        ctrl.reset()
        self.reset()
        x = x0s.copy()
        y = self.observe(x)
        gam = 1
        for t in range(T):
            # Step forward
            u = ctrl.control(y)
            x_ = self.step(x, u, nonoise=nonoise)
            y_ = self.observe(x_, nonoise=nonoise)
            re = (x * (self.Q @ x.T).T).sum(1) + (u * (self.R @ u.T).T).sum(1)
            # Record data
            xtraj[t] = x
            xtraj_[t] = x_
            ytraj[t] = y
            ytraj_[t] = y_
            utraj[t] = u
            rtraj[t] = re[:, None]
            # Update
            x = x_
            y = y_
            gam *= self.gamma

        # Flatten the trajectories
        xtraj = xtraj.reshape(-1, self.p, order='F')
        ytraj = ytraj.reshape(-1, self.r, order='F')
        utraj = utraj.reshape(-1, self.q, order='F')
        rtraj = rtraj.reshape(-1, 1, order='F')
        xtraj_ = xtraj_.reshape(-1, self.p, order='F')
        ytraj_ = ytraj_.reshape(-1, self.r, order='F')
        return xtraj, ytraj, utraj, rtraj, xtraj_, ytraj_

    def solve(self, T=None):
        ''' Solve the Riccati equation to find the optimal linear feedback
            controller
        Input:
            - T:        Integer, time horizon (None means infinite horizon)
        '''
        A, B, Q, R = self.A, self.B, self.Q, self.R
        if T is None:
            P = spl.solve_discrete_are(A, B, Q, R)
            K = -np.linalg.pinv(B.T @ P @ B + R) @ (B.T @ P @ A)
        else:
            P = Q
            K = [-np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A]
            for t in range(T-1):
                P = Q + K[0].T @ R @ K[0] + \
                        self.gamma * (A + B@K[0]).T @ P @ (A + B@K[0])
                K = [-np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A] + K
        return P, K

    def optctrl(self, T=None):
        _, K = self.solve(T=T)
        return controller.LinearFbController(K)


class VanillaLQREnv(LQREnv):
    ''' Vanilla LQR system without any noise '''
    def __init__(self, A, B, Q, R, gamma=0.99):
        p, q = B.shape
        super().__init__(A, B, np.eye(p), np.zeros((p,p)), np.zeros((p,p)), Q,
                         R, gamma=gamma, sigma_w=0, sigma_v=0)


##############################################################################
# Helper functions
##############################################################################

def random_vanilla_env(p, q, QRratio=10, Anorm=0.95):
    ''' Generates a random vanilla (noiseless) LQR environment
    Input:
        - p:        Integer, state dimension
        - q:        Integer, control dimension
        - QRratio:  Float, ration between state / control weight
        - Anorm:    Float, spectral radius of A
    '''
    A = np.random.randn(p, p)
    A = A / np.abs(np.linalg.eigvals(A)).max() * Anorm
    B = np.random.randn(p, q)
    Q = QRratio * np.eye(p)
    R = np.eye(q)
    return VanillaLQREnv(A, B, Q, R)
