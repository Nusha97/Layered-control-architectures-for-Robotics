"""
SYNOPSIS
    Running a large scale experiment on Trebuchet with 1000s of trajectories

DESCRIPTION
    Contains script with details of parameters to run a large scale experiment
    for the unicycle control problem

AUTHOR
    Anusha Srikanthan <sanusha@seas.upenn.edu>

VERSION
    0.0
"""

# Import files needed
from generate_data import gen_uni_training_data, ILQR, save_object, unicycle_K, forward_simulate, generate_polynomial_coeffs
import jax.numpy as onp
from helper_functions import compute_stl_cost
from mlp_jax import MLP
from generate_data import load_object
from model_learning import TrajDataset, train_model, eval_model, numpy_collate, save_checkpoint, restore_checkpoint
import numpy as np
import ruamel.yaml as yaml
import optax
from flax.training import train_state
import torch.utils.data as data
import jax
import matplotlib.pyplot as plt
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_affine_set
from functools import partial
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, "../tracking")

from trajgen import quadratic

gamma = 1

# Generate training data
def data_generation(num_iter, file_path):
    """
    Generates trajectory data for the unicycle model using ILQR given the number of trajectories to generate
    :param num_iter: Number of trajectories to be generated
    :param file_path: File to store the generated trajectories
    :return: actual_traj, ref_traj, rdot_traj, dynamics
    """
    # Create an ilqr object instance
    uni_ilqr1 = ILQR(unicycle_K, maxiter=1000)

    # Generate training data from ilqr
    xtraj, rtraj, rdottraj, costs, dynamics = gen_uni_training_data(uni_ilqr1, num_iter, file_path, 6, 3)

    # Save data as pickle file
    save_object([xtraj, rtraj, rdottraj, costs], file_path)

    # Load data into an object
    unicycle_data = load_object(file_path)

    # Convert the data into ref, actual and input trajectories
    actual_traj = np.vstack(unicycle_data[0])
    ref_traj = np.vstack(unicycle_data[1])
    rdot_traj = np.vstack(unicycle_data[2])

    # Return actual, ref, input, discretized dynamics
    return actual_traj, ref_traj, rdot_traj, dynamics


# Create augmented states and the dataset
def make_dataset(ref_traj, actual_traj, rdot_traj, rho=1, horizon=10):
    """
    Create the dataset using TrajDataset class with (aug_state, input, cost, next aug_state)
    :param ref_traj: List of all reference trajectories
    :param actual_traj: List of all system rollout trajectories using discretized dynamics
    :param rdot_traj: List of all input trajectories
    :param rho: Tracking penalty factor
    :param: horizon: length of each trajectory
    :return: dataset, aug_state
    """
    # Input size
    q = 3

    # Augmented state size
    p = 3 + 3 * horizon

    traj_len = ref_traj.shape[0]
    num_iter = int(traj_len / horizon)

    # Compute the trajectory tracking cost using Monte Carlo estimates
    #cost_traj, input_traj = compute_tracking_cost(ref_traj, actual_traj, rdot_traj, horizon, rho)
    cost_traj, input_traj = compute_stl_cost(ref_traj, actual_traj, rdot_traj, horizon)

    # Define the MDP state
    aug_state = []
    for i in range(num_iter):
        r0 = ref_traj[i*horizon:(i+1)*horizon, :]
        act = actual_traj[i*horizon:(i+1)*horizon, :]
        aug_state.append(np.append(act[0, :], r0))
    aug_state = np.array(aug_state)

    # Define the length of each data point
    Tstart = 0
    Tend = aug_state.shape[0]

    # Define the dataset
    dataset = TrajDataset(aug_state[Tstart:Tend - 1, :].astype('float64'),
                          input_traj[Tstart:Tend - 1, :].astype('float64'),
                          cost_traj[Tstart:Tend - 1, None].astype('float64'),
                          aug_state[Tstart + 1:Tend, :].astype('float64'))

    # Return dataset and the augmented states
    return dataset, aug_state


def load_model(p, rho, file_path):
    """
    Creates an MLP using JAX libraries and loads the model weights from a saved file
    :param p: size of augmented states
    :param rho: tracking penalty factor
    :param file_path: path to model weights
    :return: model, batch_size, model_state, num_epochs, model_save
    """

    # Load hyperparams of neural network model
    with open(file_path) as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)

    num_hidden = yaml_data['num_hidden']
    batch_size = yaml_data['batch_size']
    learning_rate = yaml_data['learning_rate']
    num_epochs = yaml_data['num_epochs']
    model_save = yaml_data['save_path']+str(rho)

    # Define the MLP
    model = MLP(num_hidden=num_hidden, num_outputs=1)

    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (batch_size, p))  # Batch size batch_size, input size p

    # Initialize the model
    params = model.init(init_rng, inp)
    #optimizer = optax.adam(learning_rate=learning_rate)
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=optimizer)

    return model, batch_size, model_state, num_epochs, model_save


def polynomial_tests(num_inf, N, order):
    """
    Generate polynomial trajectories given the number of them to be generated and the order of the polynomials
    :param num_inf: Number of trajectories to generate
    :param N: Horizon length of each trajectory
    :param order: Order of the polynomial
    :return: poly_traj, poly_aug_state, rdot_poly
    """
    inits = np.random.uniform(0, 1, (2, num_inf))
    inits = np.append(inits, np.zeros(num_inf))
    inits = np.reshape(inits, (3, num_inf))

    goals = np.random.uniform(2, 3, (2, num_inf))
    goals = np.append(goals, np.zeros(num_inf))
    goals = np.reshape(goals, (3, num_inf))

    poly_traj = []
    rdot_poly = []
    for i in range(num_inf):
        x = np.zeros([N, inits.shape[0]])
        rdot = np.zeros([N, inits.shape[0]])
        for j in range(inits.shape[0]):
            coeffs = generate_polynomial_coeffs(inits[j, i], goals[j, i], N, order)
            x[:, j] = np.polyval(coeffs, np.linspace(0, 1, N))
            t = np.poly1d(coeffs).deriv()
            rdot_coeffs = t.coefficients
            rdot[:, j] = np.polyval(rdot_coeffs, np.linspace(0, 1, N))
        poly_traj.append(x)
        rdot_poly.append(rdot)

    poly_aug_state = [np.append(poly_traj[r][0, :], poly_traj[r]) for r in range(len(poly_traj))]
    poly_aug_state = np.array(poly_aug_state)

    return poly_traj, poly_aug_state, rdot_poly


def sampler(poly, T, ts):
    """
    Function to generate samples given polynomials
    :param poly: polynomial object
    :param T: total number of time steps to sample
    :param ts: waypoint times
    :return: ref
    """
    k = 0
    ref = []
    for i, tt in enumerate(np.linspace(ts[0], ts[-1], T)):
        if tt > ts[k + 1]: k += 1
        ref.append(poly[k](tt - ts[k]))
    return ref


def compute_coeff_deriv(coeff, segments):
    """
    Function to compute the nth derivative of a polynomial
    :param coeff: polynomial coefficients
    :param segments: number of segments
    :return: coeff_new
    """
    coeff_new = coeff.copy()
    for i in range(segments):  # piecewise polynomial
        t = np.poly1d(coeff_new[i, :]).deriv()
        coeff_new[i, 0] = 0
        coeff_new[i, 1:] = t.coefficients
    return coeff_new


def compute_deriv(coeffs, Tref, segments, ts):
    """
    Compute the derivative of polynomial coeff to get xdot, ydot, yawdot
    :param coeffs: polynomial coeffs of x, y, yaw (heading angle)
    :param Tref: Total horizon of trajectory
    :param segments: number of segments
    :param ts: waypoint times
    :return: xdot_ref, ydot_ref, yawdot_ref
    """
    coeff_x = np.vstack(coeffs[0, :, :])
    coeff_y = np.vstack(coeffs[1, :, :])
    coeff_yaw = np.vstack(coeffs[2, :, :])

    dot_x = compute_coeff_deriv(coeff_x, segments)
    xdot_ref = [np.poly1d(dot_x[i, :]) for i in range(segments)]
    xdot_ref = np.vstack(sampler(xdot_ref, Tref, ts)).flatten()

    dot_y = compute_coeff_deriv(coeff_y, segments)
    ydot_ref = [np.poly1d(dot_y[i, :]) for i in range(segments)]
    ydot_ref = np.vstack(sampler(ydot_ref, Tref, ts)).flatten()

    dot_yaw = compute_coeff_deriv(coeff_yaw, segments)
    yawdot_ref = [np.poly1d(dot_yaw[i, :]) for i in range(segments)]
    yawdot_ref = np.vstack(sampler(yawdot_ref, Tref, ts)).flatten()

    return np.reshape([xdot_ref, ydot_ref, yawdot_ref], (Tref * 3), order='C')


def polynomial_wp_tests(num_inf, N, order):
    """
    Generate polynomial trajectories of given order between waypoints
    :param num_inf: Number of trajectories to be generated
    :param N: Horizon length of each trajectory
    :param order: order of polynomial
    :return: poly_traj, poly_aug_state, rdot_poly
    """
    inits = np.random.uniform(0, 2, (2, num_inf))
    inits = np.append(inits, np.zeros(num_inf))
    inits = np.reshape(inits, (3, num_inf))
    print(inits)

    goals = np.random.uniform(1, 3, (2, num_inf))
    goals = np.append(goals, np.zeros(num_inf))
    goals = np.reshape(goals, (3, num_inf))
    print(goals)

    poly_traj = []
    rdot_poly = []
    for i in range(num_inf):
        wp = np.zeros((3, 3))
        wp[0, :] = inits[:, i]
        wp[1, :] = inits[:, i]*2/3+goals[:, i]/2
        wp[2, :] = goals[:, i]
        ts = np.linspace(0, 1, len(wp))

        poly_ref, poly_coeff = quadratic.generate(wp, ts, order, N, 3, None, 0)
        rdot_poly.append(compute_deriv(poly_coeff, N-1, len(wp)-1, ts))
        poly_traj.append(poly_ref)

    poly_aug_state = [np.append(inits[:, r], poly_traj[r]) for r in range(len(poly_traj))]
    poly_aug_state = onp.array(poly_aug_state)

    return poly_traj, poly_aug_state, rdot_poly


def test_opt(trained_model_state, value, aug_test_state, ts, N, num_inf, dynamics, rho):
    """
    Planner method that includes the learned tracking cost function and the utility cost
    :param trained_model_state: weights of the trained model
    :param value: parameter to tradeoff between utility cost and tracking cost
    :param aug_test_state: Test trajectories
    :param ts: waypoint times
    :param N: horizon of each trajectory
    :param num_inf: total number of test trajectories
    :param dynamics: discretized dynamics of unicycle model for forward simulation
    :param rho: tracking penalty factor
    :return: sim_cost, init_cost
    """
    def calc_cost_GD(wp, ref):
        pred = trained_model_state.apply_fn(trained_model_state.params, ref).ravel()
        wp_cost = onp.array(wp - ref[3 * (ts[1]):3 * (ts[1] + 1)])
        return value * onp.exp(pred[0]) + onp.sum(wp_cost)

    A = np.zeros((6, (N+1)*3))
    A[0, 0] = 1
    A[1, 1] = 1
    A[2, 2] = 1
    A[-3, -3] = 1
    A[-2, -2] = 1
    A[-1, -1] = 1

    solution = []
    sim_cost = []
    init_cost = []
    ref = []
    rollout = []
    times = []

    for i in range(num_inf):
        init = aug_test_state[i, 0:3]
        goal = aug_test_state[i, -3:]

        b = np.append(init, goal)

        init_ref = aug_test_state[i, :].copy()

        wp = init_ref[3*ts[1]:3*(ts[1]+1)]
        init_val = calc_cost_GD(wp, init_ref)

        pg = ProjectedGradient(partial(calc_cost_GD, wp), projection=projection_affine_set, maxiter=1)
        solution.append(pg.run(aug_test_state[i, :], hyperparams_proj=(A, b)))
        prev_val = init_val
        val = solution[i].state.error
        cur_sol = solution[i]
        chosen_val = val

        if rho < 1:
            loop_val = 5
        else:
            loop_val = 20
        for j in range(loop_val):
            next_sol = pg.update(cur_sol.params, cur_sol.state, hyperparams_proj=(A, b))
            val = next_sol.state.error
            if val < prev_val:
                chosen_val = val
                solution[i] = cur_sol
            prev_val = val
            cur_sol = next_sol

        sol = solution[i]
        new_aug_state = sol.params
        if chosen_val > init_val:
            new_aug_state = init_ref

        x0 = init_ref[0:3]
        ref.append(new_aug_state[3:].reshape([N, 3]))

        c, x = forward_simulate(dynamics, x0, ref[i], None, N)
        sim_cost.append(c)
        rollout.append(x)
        x0 = aug_test_state[i, 0:3]
        ci, xi = forward_simulate(dynamics, x0, aug_test_state[i, 3:].reshape([N,3]), None, N)
        init_cost.append(ci)

    return sim_cost, init_cost


def main():
    """
    Main function to run the experiment for evaluation of the neural network models
    :return: None
    """
    num_iter = 500
    horizon = 10

    p = 3 + 3*horizon

    rho = 1

    uni_ilqr1 = ILQR(unicycle_K, maxiter=1000)
    dynamics = uni_ilqr1.discretize('rk4')

    file_path = r"./data/unicycle_train_wp.pkl"

    #ref_traj, actual_traj, rdot_traj, dynamics = data_generation(num_iter, file_path)

    #np.random.seed(7859)
    #poly_traj, poly_aug_state, rdot_poly_traj = polynomial_wp_tests(num_iter, horizon, order=5)

    #act_poly_traj = []
    #for i in range(num_iter):
    #    c, x = forward_simulate(dynamics, poly_traj[i][:3], np.reshape(poly_traj[i], (horizon, 3)),
    #                            np.reshape(rdot_poly_traj[i], (horizon-1, 3)), horizon)
    #    act_poly_traj.append(x)

    #uni_poly_data = []
    #uni_poly_data.append([act_poly_traj, poly_traj, rdot_poly_traj])

    #save_object(uni_poly_data, r"./data/unicycle_poly-stl.pkl")

    unicycle_data = load_object(file_path)
    uni_poly_data = load_object(r"./data/unicycle_poly-v7.pkl")


    actual_traj = np.append(unicycle_data[0], uni_poly_data[0][0])
    actual_traj = np.reshape(actual_traj, [num_iter * horizon * 2, 3], order='F')

    ref_traj = np.append(unicycle_data[1], uni_poly_data[0][1])
    ref_traj = np.reshape(ref_traj, [num_iter * horizon * 2, 3], order='F')

    rdot_traj = np.append(unicycle_data[2], uni_poly_data[0][2])
    rdot_traj = np.reshape(rdot_traj, [num_iter * (horizon-1) * 2, 3], order='F')

    print(rdot_traj.shape)

    train_dataset, aug_state = make_dataset(ref_traj, actual_traj, rdot_traj, rho, horizon)

    model, batch_size, model_state, num_epochs, model_save = load_model(p, rho, r"./data/params.yaml")

    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
    trained_model_state = train_model(model_state, train_data_loader, num_epochs=num_epochs)

    #save_checkpoint(trained_model_state, model_save, 4)

    eval_model(trained_model_state, train_data_loader, batch_size)

    #trained_model_state = restore_checkpoint(model_state, model_save)

    num_iter = 50
    file_path = r"./data/unicycle_train_new.pkl"
    unicycle_data = load_object(file_path)
    uni_poly_data = load_object(r"./data/unicycle_poly-stl.pkl")
    actual_traj = np.append(unicycle_data[0], uni_poly_data[0][0])
    actual_traj = np.reshape(actual_traj, [num_iter * horizon * 2, 3], order='F')

    ref_traj = np.append(unicycle_data[1], uni_poly_data[0][1])
    ref_traj = np.reshape(ref_traj, [num_iter * horizon * 2, 3], order='F')

    rdot_traj = np.append(unicycle_data[2], uni_poly_data[0][2])
    rdot_traj = np.reshape(rdot_traj, [num_iter * (horizon-1) * 2, 3], order='F')

    print(rdot_traj.shape)

    train_dataset, aug_state = make_dataset(ref_traj, actual_traj, rdot_traj, rho, horizon)

    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
    trained_model_state = train_model(trained_model_state, train_data_loader, num_epochs=num_epochs)


    trained_model = model.bind(trained_model_state.params)

    # Inference
    num_inf = 50
    #ref_traj, actual_traj, rdot_traj, dynamics = data_generation(num_iter=50, file_path=r"./data/unicycle_inference_wp.pkl")

    file_path = r"./data/unicycle_inference_wp.pkl"
    unicycle_data = load_object(file_path)

    actual_traj = np.vstack(unicycle_data[0])
    actual_traj = np.reshape(actual_traj, [num_inf * horizon, 3], order='F')
    
    ref_traj = np.vstack(unicycle_data[1])
    ref_traj = np.reshape(ref_traj, [num_inf * horizon, 3], order='F')
    
    rdot_traj = np.vstack(unicycle_data[2])
    rdot_traj = np.reshape(rdot_traj, [num_inf * (horizon-1), 3], order='F')

    test_dataset, aug_test_state = make_dataset(ref_traj, actual_traj, rdot_traj, rho, horizon)

    test_data_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)
    eval_model(trained_model_state, test_data_loader, batch_size)

    # Save plot on entire test dataset
    out = []
    true = []
    for batch in test_data_loader:
        data_input, _, cost, _ = batch
        out.append(trained_model(data_input))
        true.append(cost)

    out = np.vstack(out)
    true = np.vstack(true)

    plt.figure()
    plt.plot(out.ravel(), 'b-', label="Predictions")
    plt.plot(true.ravel(), 'r--', label="Actual")
    plt.legend()
    plt.title("MLP with JAX on hold out data")
    plt.savefig("./data/inference-stl"+str(rho)+".png")


    np.random.seed(89430)
    #poly_traj, poly_aug_state, rdot_poly_traj = polynomial_wp_tests(num_inf, horizon, order=5)

    #act_poly_traj = []
    #for i in range(num_inf):
    #    c, x = forward_simulate(dynamics, poly_traj[i][:3], np.reshape(poly_traj[i], (horizon, 3)),
    #                            np.reshape(rdot_poly_traj[i], (horizon-1, 3)), horizon)
    #    act_poly_traj.append(x)

    #uni_poly_test_data = []
    #uni_poly_test_data.append([act_poly_traj, poly_traj, rdot_poly_traj])

    #save_object(uni_poly_test_data, r"./data/unicycle_poly_test.pkl")

    uni_poly_data = load_object(r"./data/unicycle_poly_test.pkl")

    actual_traj = np.vstack(uni_poly_data[0][0])
    actual_traj = np.reshape(actual_traj, [num_inf * horizon, 3], order='F')

    ref_traj = np.vstack(uni_poly_data[0][1])
    ref_traj = np.reshape(ref_traj, [num_inf * horizon, 3], order='F')
    
    rdot_traj = np.vstack(uni_poly_data[0][2])
    rdot_traj = np.reshape(rdot_traj, [num_inf * (horizon-1), 3], order='F')

    print(rdot_traj.shape)

    test_dataset, aug_test_state = make_dataset(ref_traj, actual_traj, rdot_traj, rho, horizon)

    test_data_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)
    eval_model(trained_model_state, test_data_loader, batch_size)

    # Save plot on entire test dataset
    out = []
    true = []
    for batch in test_data_loader:
        data_input, _, cost, _ = batch
        out.append(trained_model(data_input))
        true.append(cost)

    out = np.exp(np.vstack(out))
    true = np.exp(np.vstack(true))

    plt.figure()
    plt.plot(out.ravel(), 'b-', label="Predictions")
    plt.plot(true.ravel(), 'r--', label="Actual")
    plt.legend()
    plt.title("MLP with JAX on hold out data")
    plt.savefig("./data/inference-stl-poly"+str(rho)+".png")


if __name__ == '__main__':
    main()






