"""
    Contains functions for quadrotor utility
"""

import numpy as np
from nonlinear_dynamics import *
import trajgen
import test_utils


def compute_coeff_deriv(coeff, n):
    """
    Function to compute the nth derivative of a polynomial
    :return:
    """
    num_piecewise, p = coeff.shape # number of waypoints, order of polynomial
    for i in range(len(num_piecewise)-1): # piecewise polynomial
        for j in range(n): # Compute nth derivative of polynomial
            t = np.poly1d([coeff[i, :]])
            t = t.deriv()
            coeff[i, j] = 0
            coeff[i, j+1:] = t.coefficients
    return coeff


def sampler(poly):
    """
    Function to generate samples given polynomials
    :param coeff:
    :return:
    """
    points = np.array([np.linspace(0, 1, T) ** (i + 1) for i in range(order)])
    return poly(points)


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
                    [[0.0,1.0,0.0,0.0],
                    [0.0,0.0,g,0.0],
                    [0.0,0.0,0.0,1.0],
                    [0.0,0.0,0.0,0.0]])
    Bx = np.array(
                    [[0.0],
                    [0.0],
                    [0.0],
                    [np.sin(t1)*l/Ixx]])
    Ay = np.array(
                    [[0.0,1.0,0.0,0.0],
                    [0.0,0.0,-1.0*g,0.0],
                    [0.0,0.0,0.0,1.0],
                    [0.0,0.0,0.0,0.0]])
    By = np.array(
                    [[0.0],
                    [0.0],
                    [0.0],
                    [np.sin(t1)*l/Iyy]])
    Az = np.array(
                    [[0.0,1.0],
                    [0.0,0.0]])
    Bz = np.array(
                    [[0.0],
                    [1.0/m]])
    Ayaw = np.array(
                    [[0.0,1.0],
                    [0.0,0.0]])
    Byaw = np.array(
                    [[0.0],
                    [Ktao/(Kt*Izz)]])


    def __init__(self, dist):
        self.dist = dist
        self.coeff_x = None
        self.coeff_y = None
        self.coeff_z = None
        self.coeff_yaw = None


    def get_zb(self, p, num_piecewise):
        """
        Function to compute
        :return:
        """
        ddot_coeff = []
        ddot_coeff.append(compute_coeff_deriv(self.coeff_x, 2))
        ddot_coeff.append(compute_coeff_deriv(self.coeff_y, 2))
        ddot_coeff.append(compute_coeff_deriv(self.coeff_z, 2))

        # Sample ref trajectories
        ddot_x = np.poly1d(ddot_coeff[0])
        ddot_y = np.poly1d(ddot_coeff[1])
        ddot_z = np.poly1d(ddot_coeff[2])

        Tref = 25

        ddot_ref = np.zeros([3, Tref+1])
        ddot_ref[0, :] = sampler(ddot_x)
        ddot_ref[1, :] = sampler(ddot_y)
        ddot_ref[2, :] = sampler(ddot_z) + g*np.ones(Tref+1)

        return  ddot_ref/np.linalg.norm(ddot_ref, axis=0) # Should this be computed for each waypoint separately?


    def get_xb(self):
        """

        :return:
        """
        x = np.cross(self.get_yc(), self.get_zb())
        return x/np.linalg.norm(x)


    def get_yb(self, p, num_piecewise):
        """

        :return:
        """
        return np.cross(self.get_zb(p, num_piecewise), self.get_xb()) # For each time step has to be done


    def get_yc(self, p, num_piecewise):
        """

        :return:
        """
        return np.array([[np.sin(self.coeff_yaw)], [np.cos(self.coeff_yaw)], np.zeros(p*num_piecewise)])


    def get_hw(self):
        """

        :param self:
        :return:
        """


    def intermediate_qt(self):
        """
        Function to compute intermediaries for going from flat outputs to states
        :return:
        """


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
            self.coeff_x = coeff_x
            self.coeff_y = coeff_y
            self.coeff_z = coeff_z
            self.coeff_yaw = coeff_yaw
            # Isolate outputs
            # Call intermediate qt
            # Compute full state



    def disturbance(self, dist):
        """
        Function to introduce disturbances in your dynamics
        :param dist:
        :return:
        """
        if dist is None:
            pass
        else:
            return



class non_linear_quad:
    """
        Nonlinear dynamics of quadrotor with options
    """
    def __init__(self):
        """
            Initialize full DOF nonlinear quad
        """
