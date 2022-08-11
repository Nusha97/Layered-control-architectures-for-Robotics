import numpy as np
import quadrotortrajectorygen as qt
from test_utils import *

# Decentralized linearized control
# X-subsystem
# The state variables are x, dot_x, pitch, dot_pitch

g = 9.80
m = 1

Ix = 8.1 * 1e-3
Iy = 8.1 * 1e-3
Iz = 14.2 * 1e-3

Ax = np.array([[0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, g, 0.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 0.0]])

Bx = np.array([[0.0], [0.0], [0.0], [1 / Ix]])

# Y-subsystem
# The state variables are y, dot_y, roll, dot_roll
Ay = np.array([[0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, -g, 0.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 0.0]])

By = np.array([[0.0],
      [0.0],
      [0.0],
      [1 / Iy]])

# Z-subsystem
# The state variables are z, dot_z
Az = np.array([[0.0, 1.0],
      [0.0, 0.0]])
Bz = np.array([[0.0],
      [1 / m]])

# Yaw-subsystem
# The state variables are yaw, dot_yaw
Ayaw = np.array([[0.0, 1.0],
        [0.0, 0.0]])
Byaw = np.array([[0.0],
        [1 / Iz]])

def compute_reward():
    """
    Function to compute the reward at each time step for the trajectory
    :return:
    """



def compute_xtraj_motion_prim(traj, Tf):
    """
    Compute quadrotor trajectories using motion primitives
    :return:
    """
    numPlotPoints = 100
    time = np.linspace(0, Tf, numPlotPoints)
    position = np.zeros([numPlotPoints, 3])
    velocity = np.zeros([numPlotPoints, 3])
    acceleration = np.zeros([numPlotPoints, 3])
    thrust = np.zeros([numPlotPoints, 1])
    ratesMagn = np.zeros([numPlotPoints, 1])

    for i in range(numPlotPoints):
        t = time[i]
        position[i, :] = traj.get_position(t)
        velocity[i, :] = traj.get_velocity(t)
        acceleration[i, :] = traj.get_acceleration(t)
        thrust[i] = traj.get_thrust(t)
        ratesMagn[i] = np.linalg.norm(traj.get_body_rates(t))

    return [position, velocity, acceleration], [thrust, ratesMagn]
