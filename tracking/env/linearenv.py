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


class linearquad(LQREnv):
    """
    Author: Anusha Srikanthan
    A quad-rotor object class based on linearized quad-rotor subsystems with
    functions to compute transformation from flat outputs to states and perform
    LQR control on the subsystems
    """
    # define constants
    g = 9.81  # m/s2
    b = 0.01  # air drag/friction force
    c = 0.2  # air friction constant

    # quadrotor physical constants
    m = 1.0  # kg  mass of the quadrotor
    Ixx = 0.5  # kg*m2 moment of inertia around X-axis (quadrotor rotates around X-axis)
    Iyy = 0.5  # kg*m2
    Izz = 0.5  # kg*m2
    Ktao = 0.02  # Drag torque constant for motors
    Kt = 0.2  # Thrust constant for motors

    # quadrotor geometry constants
    t1 = np.pi / 4  # rads
    t2 = 3 * np.pi / 4  # rads
    t3 = 5 * np.pi / 4  # rads
    t4 = 7 * np.pi / 4  # rads
    l = 0.2  # m  arm length from center of mass to each rotor

    ## Ignore these systems for now
    Ax = np.array(
        [[0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, g, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 0.0]])
    Bx = np.array(
        [[0.0, 1],
         [0.0, 1],
         [0.0, 1],
         [np.sin(t1) * l / Ixx, 1]])
    Ay = np.array(
        [[0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0 * g, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 0.0]])
    By = np.array(
        [[0.0],
         [0.0],
         [0.0],
         [np.sin(t1) * l / Iyy]])
    Az = np.array(
        [[0.0, 1.0],
         [0.0, 0.0]])
    Bz = np.array(
        [[0.0],
         [1.0 / m]])
    Ayaw = np.array(
        [[0.0, 1.0],
         [0.0, 0.0]])
    Byaw = np.array(
        [[0.0],
         [Ktao / (Kt * Izz)]])

    def __init__(self, dist):
        self.dist = dist
        self.coeff_x = None
        self.coeff_y = None
        self.coeff_z = None
        self.coeff_yaw = None
        self.waypt = None
        self.Tref = None

    def get_T(self, num_waypt):
        """

        :param Tref:
        :param num_waypt:
        :return:
        """
        ddot_coeff = []
        ddot_coeff.append(compute_coeff_deriv(self.coeff_x, 2, num_waypt))
        ddot_coeff.append(compute_coeff_deriv(self.coeff_y, 2, num_waypt))
        ddot_coeff.append(compute_coeff_deriv(self.coeff_z, 2, num_waypt))

        # Sample ref trajectories
        ddot_x = [np.poly1d(ddot_coeff[0][i, :]) for i in range(num_waypt)]  # x
        ddot_y = [np.poly1d(ddot_coeff[1][i, :]) for i in range(num_waypt)]  # y
        ddot_z = [np.poly1d(ddot_coeff[2][i, :]) for i in range(num_waypt)]  # z

        ddot_ref = []
        ddot_ref.append(sampler(ddot_x, self.Tref, num_waypt, self.waypt))
        ddot_ref.append(sampler(ddot_y, self.Tref, num_waypt, self.waypt))
        ddot_ref.append(sampler(ddot_z, self.Tref, num_waypt, self.waypt) + g * np.ones([self.Tref]))

        ddot_ref = np.vstack(ddot_ref).flatten()
        ddot_ref = np.reshape(ddot_ref, [3, self.Tref], order='C')
        return ddot_ref

    def get_zb(self, ddot_ref):
        """
        Function to compute
        :return:
        """
        return (ddot_ref / np.linalg.norm(ddot_ref, axis=0)).T

    def get_xb(self, yc, zb):
        """

        :return:
        """
        x = []
        for y, z in zip(yc, zb):
            x.append(np.cross(y.flatten(), z.flatten()))
        return np.vstack(x) / np.linalg.norm(np.vstack(x))

    def get_yb(self, zb, xb):
        """

        :return:
        """
        r = []
        for z, x in zip(zb, xb):
            r.append(np.cross(z.flatten(), x.flatten()))
        return np.vstack(r)  # For each time step has to be done

    def get_yc(self, num_waypt):
        """

        :return:
        """
        yaw = [np.poly1d(self.coeff_yaw[i, :].value) for i in range(num_waypt)]
        ref = sampler(yaw, self.Tref, num_waypt, self.waypt)
        ref = np.vstack(ref)
        temp = np.stack([-np.sin(ref), np.cos(ref), np.zeros([self.Tref, 1])]).flatten()
        temp = temp.reshape((3, self.Tref))
        return temp.T
        # return np.stack([-np.sin(ref), np.cos(ref), np.zeros([num_waypt, Tref])]).flatten()

    def get_hw(self, ddot_ref, num_waypt, zb):
        """

        :param self:
        :return:
        """
        dddot_coeff = []
        dddot_coeff.append(compute_coeff_deriv(self.coeff_x, 3, num_waypt))
        dddot_coeff.append(compute_coeff_deriv(self.coeff_y, 3, num_waypt))
        dddot_coeff.append(compute_coeff_deriv(self.coeff_z, 3, num_waypt))

        # Sample ref trajectories
        dddot_x = [np.poly1d(dddot_coeff[0][i, :]) for i in range(num_waypt)]
        dddot_y = [np.poly1d(dddot_coeff[1][i, :]) for i in range(num_waypt)]
        dddot_z = [np.poly1d(dddot_coeff[2][i, :]) for i in range(num_waypt)]

        dddot_ref = []
        dddot_ref.append(sampler(dddot_x, self.Tref, num_waypt, self.waypt))
        dddot_ref.append(sampler(dddot_y, self.Tref, num_waypt, self.waypt))
        dddot_ref.append(sampler(dddot_z, self.Tref, num_waypt, self.waypt))

        dddot_ref = np.vstack(dddot_ref).flatten()
        dddot_ref = np.reshape(dddot_ref, [3, self.Tref], order='C')
        prod = []
        for a, b, c in zip(dddot_ref.T, zb, zb.T):
            prod.append(a @ b * c)
        return (dddot_ref - np.vstack(prod)) / np.linalg.norm(ddot_ref, axis=0)

    def intermediate_qt(self, num_waypt):
        """
        Function to compute intermediaries for going from flat outputs to states
        :return:
        """
        T = self.get_T(num_waypt)
        zb = self.get_zb(T)  # 2D array of size 3 x (num_waypt * Tref)
        yc = self.get_yc(num_waypt)
        xb = self.get_xb(yc, zb)
        yb = self.get_yb(zb, xb)
        hw = self.get_hw(T, num_waypt, zb)

        return [xb, yb, zb, yc, T, hw]

    def compute_states(self, coeff_x, coeff_y, coeff_z, coeff_yaw, waypt, Tref):
        """
        Function takes in reference trajectories of flat outputs and computes
        the reference trajectories for quadrotor states
        :param ref: reference trajectory generated on flat outputs
        :return: x_traj
        """
        if coeff_x is None or coeff_y is None or coeff_z is None or coeff_yaw is None:
            return "No reference polynomial coeff provided"

        else:
            self.coeff_x = coeff_x  # 2D arrays of size num_waypt, order of polynomial
            self.coeff_y = coeff_y
            self.coeff_z = coeff_z
            self.coeff_yaw = coeff_yaw
            self.waypt = waypt
            self.Tref = Tref

            # Isolate outputs
            ts = np.array(self.waypt)
            durations = ts[1:] - ts[:-1]
            num_waypt, order = self.coeff_x.shape

            # Call intermediate qt
            xb, yb, zb, yc, T, hw = self.intermediate_qt(num_waypt)
            e3 = np.array([0, 0, 1])
            zw = np.tile(e3, [num_waypt, self.Tref]).T

            # Compute full state
            x_ref = [np.poly1d(self.coeff_x[i, :].value) for i in range(num_waypt)]
            x_ref = np.vstack(sampler(x_ref, self.Tref, num_waypt, self.waypt)).flatten()

            dot_x = compute_coeff_deriv(self.coeff_x, 1, num_waypt)
            xdot_ref = [np.poly1d(dot_x[i, :]) for i in range(num_waypt)]
            xdot_ref = np.vstack(sampler(xdot_ref, self.Tref, num_waypt, self.waypt)).flatten()

            y_ref = [np.poly1d(self.coeff_y[i, :].value) for i in range(num_waypt)]
            y_ref = np.vstack(sampler(y_ref, self.Tref, num_waypt, self.waypt)).flatten()

            dot_y = compute_coeff_deriv(self.coeff_y, 1, num_waypt)
            ydot_ref = [np.poly1d(dot_y[i, :]) for i in range(num_waypt)]
            ydot_ref = np.vstack(sampler(ydot_ref, self.Tref, num_waypt, self.waypt)).flatten()

            z_ref = [np.poly1d(self.coeff_z[i, :].value) for i in range(num_waypt)]
            z_ref = np.vstack(sampler(z_ref, self.Tref, num_waypt, self.waypt)).flatten()

            dot_z = compute_coeff_deriv(self.coeff_z, 1, num_waypt)
            zdot_ref = [np.poly1d(dot_z[i, :]) for i in range(num_waypt)]
            zdot_ref = np.vstack(sampler(zdot_ref, self.Tref, num_waypt, self.waypt)).flatten()

            # Introduce temp var
            prod1 = []
            prod2 = []
            for z, y, x in zip(zw, yb, xb):
                prod1.append(z @ y)
                prod2.append(z @ x)
            roll_ref = np.arcsin(np.vstack(prod1) * zb / (np.cos(np.arcsin(np.vstack(prod2)))))  # phi

            print("Roll", roll_ref.shape)
            # roll_ref = np.reshape(roll_ref, [Tref])
            prod1 = []
            prod2 = []
            prod3 = []
            for y, h, z, x in zip(yb, hw.T, zw, xb):
                prod1.append(-y @ h)
                prod2.append(z @ x)
                prod3.append(x @ h)
            rolldot_ref = np.vstack(prod1).flatten()

            pitch_ref = -np.arcsin(np.vstack(prod2)).flatten()  # theta
            # pitch_ref = np.reshape(pitch_ref, [num_waypt, Tref])

            pitchdot_ref = np.vstack(prod3).flatten()

            yaw_ref = [np.poly1d(self.coeff_yaw[i, :].value) for i in range(num_waypt)]
            yaw_ref = np.vstack(sampler(yaw_ref, self.Tref, num_waypt, self.waypt)).flatten()

            dot_yaw = compute_coeff_deriv(self.coeff_yaw, 1, num_waypt)
            yawdot_ref = [np.poly1d(dot_yaw[i, :]) for i in range(num_waypt)]
            yawdot_ref = np.vstack(sampler(yawdot_ref, self.Tref, num_waypt, self.waypt)).flatten()

            return [x_ref, xdot_ref, y_ref, ydot_ref, z_ref, zdot_ref, roll_ref, rolldot_ref, pitch_ref, pitchdot_ref,
                    yaw_ref, yawdot_ref]



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


def compute_coeff_deriv(coeff, n, num_waypt):
    """
    Author: Anusha Srikanthan
    Function to compute the nth derivative of a polynomial
    :return:
    """
    coeff_new = coeff.value.copy()
    for i in range(num_waypt):  # piecewise polynomial
        for j in range(n):  # Compute nth derivative of polynomial
            t = np.poly1d(coeff_new[i, :]).deriv()
            coeff_new[i, j] = 0
            coeff_new[i, j+1:] = t.coefficients
    return coeff_new


def sampler(poly, T, num_waypt, ts):
    """
    Author: Anusha Srikanthan
    Function to generate samples given polynomials
    :param coeff:
    :return:
    """
    k = 0
    ref = []
    for i, tt in enumerate(np.linspace(ts[0], ts[-1], T)):
        if tt > ts[k + 1]: k += 1
        ref.append(poly[k](tt-ts[k]))
    # return [poly[i](np.linspace(0, 1, T)) for i in range(num_waypt)]
    return ref
