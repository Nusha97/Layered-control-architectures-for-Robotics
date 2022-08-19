########################################################
"""
    Code to generate quadrotor trajectories for x, y, z, yaw
    given LQR controller parameters
    TBD: To add functionality to sample yaw even when not
    specified in the reference trajectory
"""


# Quadrotor system dynamics

import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt

import lstd
import test_utils
import trajgen

# define constants
g = 9.81 #m/s2
b = 0.01  # air drag/friction force
c = 0.2 #air friction constant

# quadrotor physical constants
m = 1.0  #kg  mass of the quadrotor
Ixx = 0.5   # kg*m2 moment of
# inertia around X-axis (quadrotor rotates around X-axis)
Iyy = 0.5   # kg*m2
Izz = 0.5   # kg*m2
Ktao = 0.02           # Drag torque constant for motors
Kt = 0.2             # Thrust constant for motors


# quadrotor geometry constants
t1 = np.pi/4   # rads
t2 = 3*np.pi/4  #rads
t3 = 5*np.pi/4 # rads
t4 = 7*np.pi/4 # rads
l = 0.2 #m  arm lenght from center of mass to each rotor


# X  axis dynamics
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

# X position output matrix
Cx = np.array([1.0,0.0,0.0,0.0])
# X velocity output matrix
Cx_dot = np.array([0.0,1.0,0.0,0.0])
# Pitch angle output matrix
Cp = np.array([0.0,0.0,1.0,0.0])
# Pitch angle velocity output matrix
Cp_dot = np.array([0.0,0.0,0.0,1.0])


# Y  dynamics
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

# Y position output matrix
Cy = np.array([1.0,0.0,0.0,0.0])
# Y velocity output matrix
Cy_dot = np.array([0.0,1.0,0.0,0.0])
# Roll angle output matrix
Cr = np.array([0.0,0.0,1.0,0.0])
# Roll angle velocity output matrix
Cr_dot = np.array([0.0,0.0,0.0,1.0])


# Z axis dynamics
Az = np.array(
[[0.0,1.0],
[0.0,0.0]])

Bz = np.array(
[[0.0],
[1.0/m]])

# Z position output matrix
Cz = np.array([1.0,0.0])
# Z velocity matrix
Cz_dot = np.array([0.0,1.0])

# Yaw dynamics
Ayaw = np.array(
[[0.0,1.0],
[0.0,0.0]])

Byaw = np.array(
[[0.0],
[Ktao/(Kt*Izz)]]) 

# Yaw angle output matrix
Cyaw = np.array([1.0,0.0])
# Yaw angle velocity output matrix
Cyaw_dot = np.array([0.0,1.0])

# Transmission matrix
D = np.array([0.0])

# Generating x, y, z, yaw waypoints - Can skip yaw

waypoints = [[0, 0, 0, 0],
             [np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()],
             [np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()],
             [1, 1, 1, 1]]
waypoints = np.array(waypoints)
ts = [0, 0.33, 0.66, 1]
num_points = 20
tp = np.linspace(0, 1, 26)

# LQR Controller for x-subsystem

p = 4
q = 1

sigma = 0

Qx, Rx = 100*np.zeros([p, p]), np.eye(q)
Qx[0, 0] = 1
Tref = 25
# Construct augmented system for tracking
At, Bt, Qt, Rt = lstd.nominal_to_tracking(Ax, Bx, Qx, Rx, Tref)

Pstar = spl.solve_discrete_are(At, Bt, Qt, Rt)
Kstar = -np.linalg.pinv(Bt.T @ Pstar @ Bt + Rt) @ (Bt.T @ Pstar @ At)

static_ctrl = test_utils.linear_feedback_controller(Kstar)

# Solve for a time-varying controller using value iteration
Pvi, Kvi = test_utils.lqr_vi(At, Bt, Qt, Rt, 100, 0.99)
vi_ctrlr = test_utils.TVcontroller(Kvi[:])


# Spline trajectory
rhos = np.concatenate([[0], np.logspace(4, 8, 3)])
tp = np.linspace(0, 1, 26)

plt.figure(figsize=(20, 10))
for i, rho in enumerate(rhos):
    ref = trajgen.generate(waypoints, ts, 5, Tref+1, p, P=Pstar, rho=rho)
    d0 = np.concatenate([waypoints[0], ref])

    xtraj, utraj, rtraj = test_utils.sample_traj(At, Bt, Qt, Rt, static_ctrl, Tref, d0, sigma=0)
    vi_ctrlr = test_utils.TVcontroller(Kvi[:])
    xvi, uvi, rvi = test_utils.sample_traj(At, Bt, Qt, Rt, vi_ctrlr.ctrl, Tref, d0, sigma=0)
    
    plt.subplot(3, len(rhos), i+1+len(rhos))
    plt.plot(tp, ref[0::p], 'x-', label='reference x')
    # plt.plot(tp, xvi[:,0], '+-', label='tracking')
    plt.plot(tp, xtraj[:,0], '+-', label='static x')
    plt.plot(ts, waypoints[:, 0], 'r^', label='waypoints')
    plt.legend()
    
    plt.subplot(3, len(rhos), i+1+2*len(rhos))
    plt.plot(np.arange(Tref), np.abs(utraj), '+--', label='static')
    # plt.plot(np.arange(Tref), np.abs(uvi), '+--', label='opt')
    plt.legend()
    plt.show()
    
"""    
# LQR Controller for y-subsystem

p = 4
q = 1

sigma = 0

Qy, Ry = 100*np.eye(p), np.eye(q)
Tref = 25
# Construct augmented system for tracking
At, Bt, Qt, Rt = lstd.nominal_to_tracking(Ay, By, Qy, Ry, Tref)

Pstar = spl.solve_discrete_are(At, Bt, Qt, Rt)
Kstar = -np.linalg.pinv(Bt.T @ Pstar @ Bt + Rt) @ (Bt.T @ Pstar @ At)

static_ctrl = test_utils.linear_feedback_controller(Kstar)

# Solve for a time-varying controller using value iteration
Pvi, Kvi = test_utils.lqr_vi(At, Bt, Qt, Rt, 100, 0.99)
vi_ctrlr = test_utils.TVcontroller(Kvi[:])


plt.figure(figsize=(20, 10))
for i, rho in enumerate(rhos):
    ref = trajgen.generate(waypoints, ts, 5, Tref+1, p, P=Pstar, rho=rho)
    d0 = np.concatenate([waypoints[1], ref])

    xtraj, utraj, rtraj = test_utils.sample_traj(At, Bt, Qt, Rt, static_ctrl, Tref, d0, sigma=0)
    vi_ctrlr = test_utils.TVcontroller(Kvi[:])
    xvi, uvi, rvi = test_utils.sample_traj(At, Bt, Qt, Rt, vi_ctrlr.ctrl, Tref, d0, sigma=0)
    
    plt.subplot(3, len(rhos), i+1+len(rhos))
    plt.plot(tp, ref[0::p], 'x-', label='reference y')
    # plt.plot(tp, xvi[:,0], '+-', label='tracking')
    plt.plot(tp, xtraj[:, 0], '+-', label='static y')
    plt.plot(ts, waypoints[:, 1], 'r^', label='waypoints')
    plt.legend()
    
    plt.subplot(3, len(rhos), i+1+2*len(rhos))
    plt.plot(np.arange(Tref), np.abs(utraj), '+--', label='static')
    # plt.plot(np.arange(Tref), np.abs(uvi), '+--', label='opt')
    plt.legend()
    

# LQR Controller for z-subsystem

p = 2
q = 1

sigma = 0

Qz, Rz = 100*np.eye(p), np.eye(q)
Tref = 25
# Construct augmented system for tracking
At, Bt, Qt, Rt = lstd.nominal_to_tracking(Az, Bz, Qz, Rz, Tref)

Pstar = spl.solve_discrete_are(At, Bt, Qt, Rt)
Kstar = -np.linalg.pinv(Bt.T @ Pstar @ Bt + Rt) @ (Bt.T @ Pstar @ At)

static_ctrl = test_utils.linear_feedback_controller(Kstar)

# Solve for a time-varying controller using value iteration
Pvi, Kvi = test_utils.lqr_vi(At, Bt, Qt, Rt, 100, 0.99)
vi_ctrlr = test_utils.TVcontroller(Kvi[:])

plt.figure(figsize=(20, 10))
for i, rho in enumerate(rhos):
    ref = trajgen.generate(waypoints, ts, 5, Tref+1, p, P=Pstar, rho=rho)
    d0 = np.concatenate([waypoints[2], ref])

    xtraj, utraj, rtraj = test_utils.sample_traj(At, Bt, Qt, Rt, static_ctrl, Tref, d0, sigma=0)
    vi_ctrlr = test_utils.TVcontroller(Kvi[:])
    xvi, uvi, rvi = test_utils.sample_traj(At, Bt, Qt, Rt, vi_ctrlr.ctrl, Tref, d0, sigma=0)
    
    plt.subplot(3, len(rhos), i+1+len(rhos))
    plt.plot(tp, ref[0::p], 'x-', label='reference z')
    # plt.plot(tp, xvi[:,0], '+-', label='tracking')
    plt.plot(tp, xtraj[:, 0], '+-', label='static z')
    plt.plot(ts, waypoints[:, 2], 'r^', label='waypoints')
    plt.legend()
    
    plt.subplot(3, len(rhos), i+1+2*len(rhos))
    plt.plot(np.arange(Tref), np.abs(utraj), '+--', label='static')
    # plt.plot(np.arange(Tref), np.abs(uvi), '+--', label='opt')
    plt.legend()
    
    
# LQR Controller for yaw-subsystem - To add functionality to be able to use even when yaw not given as waypt

p = 2
q = 1

sigma = 0

Qyaw, Ryaw = 100*np.eye(p), np.eye(q)
Tref = 25
# Construct augmented system for tracking
At, Bt, Qt, Rt = lstd.nominal_to_tracking(Ayaw, Byaw, Qyaw, Ryaw, Tref)

Pstar = spl.solve_discrete_are(At, Bt, Qt, Rt)
Kstar = -np.linalg.pinv(Bt.T @ Pstar @ Bt + Rt) @ (Bt.T @ Pstar @ At)

static_ctrl = test_utils.linear_feedback_controller(Kstar)

# Solve for a time-varying controller using value iteration
Pvi, Kvi = test_utils.lqr_vi(At, Bt, Qt, Rt, 100, 0.99)
vi_ctrlr = test_utils.TVcontroller(Kvi[:])


plt.figure(figsize=(20, 10))
for i, rho in enumerate(rhos):
    ref = trajgen.generate(waypoints, ts, 5, Tref+1, p, P=Pstar, rho=rho)
    d0 = np.concatenate([waypoints[3], ref])

    xtraj, utraj, rtraj = test_utils.sample_traj(At, Bt, Qt, Rt, static_ctrl, Tref, d0, sigma=0)
    vi_ctrlr = test_utils.TVcontroller(Kvi[:])
    xvi, uvi, rvi = test_utils.sample_traj(At, Bt, Qt, Rt, vi_ctrlr.ctrl, Tref, d0, sigma=0)
    
    plt.subplot(3, len(rhos), i+1+len(rhos))
    plt.plot(tp, ref[0::p], 'x-', label='reference yaw')
    # plt.plot(tp, xvi[:,0], '+-', label='tracking')
    plt.plot(tp, xtraj[:, 0], '+-', label='static yaw')
    plt.plot(ts, waypoints[:, 3], 'r^', label='waypoints')
    plt.legend()
    
    plt.subplot(3, len(rhos), i+1+2*len(rhos))
    plt.plot(np.arange(Tref), np.abs(utraj), '+--', label='static')
    # plt.plot(np.arange(Tref), np.abs(uvi), '+--', label='opt')
    plt.legend()"""
