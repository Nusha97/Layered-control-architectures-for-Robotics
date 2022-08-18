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
    l = 0.2  # m  arm lenght from center of mass to each rotor
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
            x = coeff


    def disturbance(self, dist):
        if


class non_linear_quad:
    """
        Nonlinear dynamics of quadrotor with options
    """
    def __init__(self):
        """
            Initialize full DOF nonlinear quad
        """

    