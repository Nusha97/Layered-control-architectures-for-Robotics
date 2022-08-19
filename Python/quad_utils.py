"""
    Contains functions for quadrotor utility
"""

import numpy as np
from nonlinear_dynamics import *
import trajgen

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
        self.coeff = None


    def get_zb(self):
        """
        Function to compute
        :return:
        """
        T = compute_coeff_deriv(self.coeff, 2)
        return  T/np.linalg.norm(T)


    def get_xb(self):
        """

        :return:
        """
        x = np.cross(self.get_yc(), self.get_zb())
        return x/np.linalg.norm(x)

    def get_yb(self):
        """

        :return:
        """
        return np.cross(self.get_zb(), self.get_xb())


    def get_yc(self):
        """

        :return:
        """



    def get_hw(self):



    def intermediate_qt(self):
        """
        Function to compute intermediaries for going from flat outputs to states
        :return:
        """

    def compute_states(self, coeff):
        """
        Function takes in reference trajectories of flat outputs and computes
        the reference trajectories for quadrotor states
        :param ref: reference trajectory generated on flat outputs
        :return: x_traj
        """
        if coeff is None:
            return "No reference polynomial coeff provided"

        else:
            self.coeff = coeff
            x = coeff[0:]
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


def compute_coeff_deriv(coeff, n):
    """
    Function to compute the nth derivative of a polynomial
    :return:
    """
    waypoints, p = 
    for i in range(len(waypoints)-1):


    