"""
    Contains functions for quadrotor utility
"""

import numpy as np
from nonlinear_dynamics import *


def compute_coeff_deriv(coeff, n, num_waypt):
    """
    Function to compute the nth derivative of a polynomial
    :return:
    """
    for i in range(len(num_waypt)-1):  # piecewise polynomial
        for j in range(n):  # Compute nth derivative of polynomial
            t = np.poly1d([coeff[i, :]])
            t = t.deriv()
            coeff[i, j] = 0
            coeff[i, j+1:] = t.coefficients
    return coeff


def sampler(poly, T, num_waypt):
    """
    Function to generate samples given polynomials
    :param coeff:
    :return:
    """
    return [poly[i](np.linspace(0, 1, T)) for i in range(num_waypt)]


class linear_quad:
    """
    Define the linearized quadrotor subsystems by defining them as static class variables
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
                    [[0.0],
                     [0.0],
                     [0.0],
                     [np.sin(t1)*l/Ixx]])
    Ay = np.array(
                    [[0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0*g, 0.0],
                     [0.0, 0.0, 0.0, 1.0],
                     [0.0, 0.0, 0.0, 0.0]])
    By = np.array(
                    [[0.0],
                     [0.0],
                     [0.0],
                     [np.sin(t1)*l/Iyy]])
    Az = np.array(
                    [[0.0, 1.0],
                     [0.0, 0.0]])
    Bz = np.array(
                    [[0.0],
                     [1.0/m]])
    Ayaw = np.array(
                    [[0.0, 1.0],
                     [0.0, 0.0]])
    Byaw = np.array(
                    [[0.0],
                     [Ktao/(Kt*Izz)]])


    def __init__(self, dist):
        self.dist = dist
        self.coeff_x = None
        self.coeff_y = None
        self.coeff_z = None
        self.coeff_yaw = None


    def get_T(self, Tref, num_waypt):
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
        ddot_ref.append(sampler(ddot_x, Tref, num_waypt))
        ddot_ref.append(sampler(ddot_y, Tref, num_waypt))
        ddot_ref.append(sampler(ddot_z, Tref, num_waypt) + g * np.ones([num_waypt, Tref]))

        ddot_ref = np.vstack(ddot_ref).flatten()
        ddot_ref = np.reshape(ddot_ref, [3, num_waypt * Tref], order='C')
        return ddot_ref


    def get_zb(self, ddot_ref):
        """
        Function to compute
        :return:
        """
        return ddot_ref/np.linalg.norm(ddot_ref, axis=0)


    def get_xb(self, yc, zb):
        """

        :return:
        """
        x = np.cross(yc, zb)
        return x/np.linalg.norm(x)


    def get_yb(self, zb, xb):
        """

        :return:
        """
        return np.cross(zb, xb)  # For each time step has to be done


    def get_yc(self, Tref, num_waypt):
        """

        :return:
        """
        yaw = [np.poly1d(self.coeff_yaw[i, :]) for i in range(num_waypt)]
        ref = sampler(yaw, Tref, num_waypt)
        return np.array([[-np.sin(ref)], [np.cos(ref)], [np.zeros(Tref*num_waypt)]])


    def get_hw(self, ddot_ref, Tref, num_waypt, zb):
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
        dddot_ref.append(sampler(dddot_x, Tref, num_waypt))
        dddot_ref.append(sampler(dddot_y, Tref, num_waypt))
        dddot_ref.append(sampler(dddot_z, Tref, num_waypt))

        dddot_ref = np.vstack(dddot_ref).flatten()
        dddot_ref = np.reshape(dddot_ref, [3, num_waypt * Tref], order='C')
        return (dddot_ref - (dddot_coeff @ zb) @ zb)/np.linalg.norm(ddot_ref, axis=0)


    def intermediate_qt(self, Tref, num_waypt):
        """
        Function to compute intermediaries for going from flat outputs to states
        :return:
        """
        T = self.get_T(Tref, num_waypt)
        zb = self.get_zb(T)  # 2D array of size 3 x (num_waypt * Tref)
        yc = self.get_yc(Tref, num_waypt)
        xb = self.get_xb(yc, zb)
        yb = self.get_yb(zb, xb)
        hw = self.get_hw(T, Tref, num_waypt, zb)

        return [xb, yb, zb, yc, T, hw]


    def compute_states(self, coeff_x, coeff_y, coeff_z, coeff_yaw):
        """
        Function takes in reference trajectories of flat outputs and computes
        the reference trajectories for quadrotor states
        :param ref: reference trajectory generated on flat outputs
        :return: x_traj
        """
        if coeff_x or coeff_y or coeff_z or coeff_yaw is None:
            return "No reference polynomial coeff provided"

        else:
            self.coeff_x = coeff_x # 2D arrays of size num_waypt, order of polynomial
            self.coeff_y = coeff_y
            self.coeff_z = coeff_z
            self.coeff_yaw = coeff_yaw

            # Isolate outputs
            Tref = 25
            num_waypt, order = self.coeff_x.shape

            # Call intermediate qt
            xb, yb, zb, yc, T, hw = self.intermediate_qt(Tref, num_waypt)
            e3 = np.array([0, 0, 1])
            zw = np.tile(e3, [num_waypt*Tref]).transpose()

            # Compute full state
            x_ref = [np.poly1d(self.coeff_x[i, :]) for i in range(num_waypt)]
            dot_x = compute_coeff_deriv(self.coeff_x, 1, num_waypt)
            xdot_ref = [np.poly1d(dot_x[i, :]) for i in range(num_waypt)]
            y_ref = [np.poly1d(self.coeff_y[i, :]) for i in range(num_waypt)]
            dot_y = compute_coeff_deriv(self.coeff_y, 1, num_waypt)
            ydot_ref = [np.poly1d(dot_y[i, :]) for i in range(num_waypt)]
            z_ref = [np.poly1d(self.coeff_z[i, :]) for i in range(num_waypt)]
            dot_z = compute_coeff_deriv(self.coeff_z, 1, num_waypt)
            zdot_ref = [np.poly1d(dot_z[i, :]) for i in range(num_waypt)]
            roll_ref = np.asin((zw.transpose() @ yb)/(np.cos(np.asin(zw.transpose @ xb))))  # phi
            rolldot_ref = -yb @ hw
            pitch_ref = -np.asin(zw.transpose @ xb)  # theta
            pitchdot_ref = xb @ hw
            yaw_ref = [np.poly1d(self.coeff_yaw[i, :]) for i in range(num_waypt)]
            dot_yaw = compute_coeff_deriv(self.coeff_yaw, 1, num_waypt)
            yawdot_ref = [np.poly1d(dot_yaw[i, :]) for i in range(num_waypt)]  # zb @ yawdot @ zw ?? Check all the matmul

            return [x_ref, xdot_ref, y_ref, ydot_ref, z_ref, zdot_ref, roll_ref, rolldot_ref, pitch_ref, pitchdot_ref, yaw_ref, yawdot_ref]


    def disturbance(self, dist):
        """
        Function to introduce disturbances in your dynamics
        Should this be a part of the quad class?
        :param dist:
        :return:
        """
        if dist is None:
            pass
        else:
            return


class non_linear_quad:
    """
        Nonlinear dynamics of quadrotor with options for adding sensor drift
    """
    def __init__(self, x, u):
        """
            Initialize full DOF nonlinear quad
        """
        self.x_dot = f(x, u)




