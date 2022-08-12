import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt

import lstd
import test_utils
import trajgen

p = 2
q = 1

sigma = 0

Az = np.matrix([[0, 1], [0, 0]])
Bz = np.matrix([[0.0], [1.0]]

Q, R = 100*np.eye(p), np.eye(q)
Tref = 25

At, Bt, Qt, Rt = lstd.nominal_to_tracking(Az, Bz, Q, R, Tref)
Pstar = spl.solve_discrete_are(At, Bt, Qt, Rt)
Kstar = -np.linalg.pinv(Bt.T @ Pstar @ Bt + Rt) @ (Bt.T @ Pstar @ At)
static_opt_ctrl = test_utils.linear_feedback_controller(Kstar)

Pvi, Kvi = test_utils.lqr_vi(At, Bt, Qt, Rt, Tref)


ctrl = test_utils.random_controller(q)
num_trajs = 100
T = Tref
xtrajs, utrajs, rtrajs = [], [], []
for _ in range(num_trajs):
    ref = np.random.randn(p*(T+2))
    xtraj, utraj, rtraj = test_utils.sample_traj((At, Bt, Qt, Rt, ctrl, T, ref, sigma=sigma)
    xtrajs.append(xtraj)
    utrajs.append(utraj)
    rtrajs.append(rtraj)
traj = lstd.construct_traj_list(xtrajs, utrajs, rtrajs)

_, P = lstd.evaluate(traj, Kstar, 0.99, sigma=sigma)
print("Relative error of value matrix is {:.3f}%".format(test_utils.relerr(Pstar, P)*100))

waypoints = [[0, 0], [np.random.randn(), np.random.randn()], [np.random.randn(), np.random.randn()], [1, 1]]
waypoints = np.array(waypoints)
ts = [0, 0.33, 0.66, 1]
num_points = 20


rhos = np.concatenate([[0], np.logspace(4, 8, 4)])

plt.figure(figsize=(20,10))
for i, rho in enumerate(rhos):
    ref = trajgen.generate(waypoints, ts, 5, Tref+1, p, P=P, rho=rho)
    d0 = np.concatenate([waypoints[0], ref])
    xtraj, utraj, rtraj = test_utils.sample_traj(At, Bt, Qt, Rt, learned_ctrl, Tref, d0, sigma=0)
    vi_ctrlr = test_utils.TVcontroller(Kvi[:])
    xvi, uvi, rvi = test_utils.sample_traj(At, Bt, Qt, Rt, vi_ctrlr.ctrl, Tref, d0, sigma=0)

    plt.subplot(3, len(rhos), i+1)
    plt.plot(np.arange(
