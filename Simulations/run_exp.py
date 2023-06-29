"""
SYNOPSIS
    Running a large scale experiment on Trebuchet with 1000s of trajectories
DESCRIPTION

    Contains script with details of parameters to run a large scale experiment
    on the remote system
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.0
"""

# Import files needed
from generate_data import gen_uni_training_data, ILQR, save_object, unicycle_K, forward_simulate, generate_polynomial_trajectory
import jax.numpy as onp
from helper_functions import compute_tracking_cost
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
import time


Kp = 5 * np.array([[2, 1, 0], [0, 1, 3]])
gamma = 1

# Generate training data
def data_generation(num_iter, file_path):
    #import ipdb;ipdb.set_trace()
    uni_ilqr1 = ILQR(unicycle_K, maxiter=1000)
    xtraj, rtraj, rdottraj, costs = gen_uni_training_data(uni_ilqr1, num_iter, 6, 3)

    # Save as pickle file
    save_object([xtraj, rtraj, rdottraj, costs], file_path)

    unicycle_data = load_object(file_path)

    actual_traj = np.vstack(unicycle_data[0])
    ref_traj = np.vstack(unicycle_data[1])
    rdot_traj = np.vstack(unicycle_data[2])

    return  actual_traj, ref_traj, rdot_traj


# Create augmented states and the dataset
def make_dataset(N, ref_traj, actual_traj, rdot_traj, rho=1, horizon=101):
    q = 2
    p = 3 + 3 * N
    traj_len = ref_traj.shape[0]
    num_iter = int(traj_len / horizon)

    cost_traj, input_traj = compute_tracking_cost(ref_traj, actual_traj, rdot_traj, Kp, N, horizon, rho)

    aug_state = []
    for i in range(num_iter):
        r0 = ref_traj[i*horizon:(i+1)*horizon, :]

        act = actual_traj[i*horizon:(i+1)*horizon, :]

        aug_state.append(np.append(act[0, :], r0))

        #act_poly_traj.append(forward_simulate(poly_traj[i][0, :], poly_traj[i], N))

        #r0 = ref_traj[i * horizon:(i + 1) * horizon, :]
        #r0 = np.append(r0, r0[-1, :] * np.ones((N - 1, 3)))
        #r0 = np.reshape(r0, (horizon + N - 1, 3))
        #for j in range(horizon - N):
        #    aug_state.append(np.append(actual_traj[j, :], r0[j:j + N, :]))
    # aug_state = [np.append(actual_traj[r, :], ref_traj[r:r+N, :]) for r in range(num_iter)]
    #print(len(act_poly_traj))
    aug_state = np.array(aug_state)
    print(aug_state.shape)

    #poly_cost_traj, poly_input_traj = compute_tracking_cost(np.array(poly_traj), act_poly_traj, np.array(rdot_poly_traj), Kp, N, horizon, rho)

    #aug_state = np.vstack(aug_state, poly_aug_state)

    #input_traj = np.vstack([input_traj, poly_input_traj])

    #cost_traj = np.vstack([cost_traj, poly_cost_traj])

    Tstart = 0
    Tend = aug_state.shape[0]

    dataset = TrajDataset(aug_state[Tstart:Tend - 1, :].astype('float64'),
                          input_traj[Tstart:Tend - 1, :].astype('float64'),
                          cost_traj[Tstart:Tend - 1, None].astype('float64'),
                          aug_state[Tstart + 1:Tend, :].astype('float64'))

    return dataset, aug_state


def load_model(p, rho, file_path):
    with open(file_path) as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)


    num_hidden = yaml_data['num_hidden']
    batch_size = yaml_data['batch_size']
    learning_rate = yaml_data['learning_rate']
    num_epochs = yaml_data['num_epochs']
    model_save = yaml_data['save_path']+str(rho)

    print(model_save)

    model = MLP(num_hidden=num_hidden, num_outputs=1)

    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (batch_size, p))  # Batch size 64, input size p
    # Initialize the model
    params = model.init(init_rng, inp)

    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.9)

    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=optimizer)

    return model, batch_size, model_state, num_epochs, model_save


def polynomial_tests(num_inf, N, order):

    inits = np.random.uniform(0, 1, (2, num_inf))
    inits = np.append(inits, np.zeros(num_inf))
    inits = np.reshape(inits, (3, num_inf))

    goals = np.random.uniform(2, 3, (2, num_inf))
    goals = np.append(goals, np.zeros(num_inf))
    goals = np.reshape(goals, (3, num_inf))

    poly_traj = []
    for i in range(num_inf):
        poly_traj.append(generate_polynomial_trajectory(inits[:, i], goals[:, i], N, order))

    poly_aug_state = [np.append(poly_traj[r][0, :], poly_traj[r]) for r in range(len(poly_traj))]
    poly_aug_state = onp.array(poly_aug_state)

    return poly_traj, poly_aug_state


def test_opt(trained_model_state, value, aug_test_state, N, num_inf):
    def calc_cost_GD(init_ref, ref):
        pred = trained_model_state.apply_fn(trained_model_state.params, ref).ravel()
        return value * onp.exp(pred[0]) + onp.linalg.norm(init_ref - ref) ** 2


    A = np.zeros((6, (N+1)*3))
    A[0, 0] = 1
    A[1, 1] = 1
    A[2, 2] = 1
    A[-3, -3] = 1
    A[-2, -2] = 1
    A[-1, -1] = 1

    # plt.figure()
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

        start = time.time()
        pg = ProjectedGradient(partial(calc_cost_GD, init_ref), projection=projection_affine_set, maxiter=1)
        solution.append(pg.run(aug_test_state[i, :], hyperparams_proj=(A, b)))
        prev_val = solution[i].state.error
        cur_sol = solution[i]
        for j in range(40):
            next_sol = pg.update(cur_sol.params, cur_sol.state, hyperparams_proj=(A, b))
            val = next_sol.state.error
            # print(val)
            if val < prev_val:
                solution[i] = next_sol

            prev_val = val
            cur_sol = next_sol
        end = time.time()

        times.append(end-start)


        #plt.plot(solution[i].params[0::3], solution[i].params[1::3], 'b-', label='opt ref')
        #plt.plot(aug_test_state[i, 0::3], aug_test_state[i, 1::3], 'r-', label='ilqr ref')

        sol = solution[i]
        new_aug_state = sol.params
        x0 = new_aug_state[0:3]

        ref.append(new_aug_state[3:].reshape([N, 3]))

        c, x = forward_simulate(x0, ref[i], N)
        sim_cost.append(c)
        rollout.append(x)
        x0 = aug_test_state[i, 0:3]
        ci, xi = forward_simulate(x0, aug_test_state[i, 3:].reshape([N,3]), N)
        init_cost.append(ci)

    save_object(times, "./data/pgd_times.pkl")

    # plt.title("Testing by opt new references")
    # plt.savefig(save_fig)
    #plt.show()

    return sim_cost, init_cost





def main():

    num_iter = 1000
    N = 101
    horizon = 101

    p = 3 + 3*N

    # rhos = [0, 1, 5, 10, 20, 50, 100]
    rhos = [20]

    file_path = r"./data/unicycle_train.pkl"
    # ref_traj, actual_traj, rdot_traj = data_generation(num_iter, file_path)

    """np.random.seed(N)

    poly_traj, poly_aug_state = polynomial_tests(num_iter, N, 4)

    np.random.seed(N)
    rdot_poly_traj, _ = polynomial_tests(num_iter, N, 2)

    act_poly_traj = []
    for i in range(num_iter):
        c, x = forward_simulate(poly_traj[i][0, :], poly_traj[i], N)
        act_poly_traj.append(x) 

    uni_poly_data = []  

    uni_poly_data.append([act_poly_traj, poly_traj, rdot_poly_traj])

    print(act_poly_traj)

    save_object(uni_poly_data, r"./data/unicycle_poly-v2.pkl")"""

    unicycle_data = load_object(file_path)

    uni_poly_data = load_object(r"./data/unicycle_poly.pkl")

    print(len(uni_poly_data))

    actual_traj = np.append(unicycle_data[0], uni_poly_data[0][0])
    actual_traj = np.reshape(actual_traj, [num_iter * N * 2, 3], order='F')
    print(actual_traj[0:3, :])

    print(actual_traj.shape)
    ref_traj = np.append(unicycle_data[1], uni_poly_data[0][1])
    ref_traj = np.reshape(ref_traj, [num_iter * N * 2, 3], order='F')

    print(ref_traj.shape)

    rdot_traj = np.append(unicycle_data[2], uni_poly_data[0][2])
    rdot_traj = np.reshape(rdot_traj, [num_iter * N * 2, 3], order='F')

    print(rdot_traj.shape)

    for rho in rhos:

        train_dataset, aug_state = make_dataset(N, ref_traj, actual_traj, rdot_traj, rho, horizon)

        model, batch_size, model_state, num_epochs, model_save = load_model(p, rho, r"./data/params.yaml")

        train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
        trained_model_state = train_model(model_state, train_data_loader, num_epochs=num_epochs)

        #save_checkpoint(trained_model_state, model_save, 3)

        eval_model(trained_model_state, train_data_loader, batch_size)

        trained_model = model.bind(trained_model_state.params)

        # Inference
        # ref_traj, actual_traj, rdot_traj = data_generation(num_iter=100, file_path=r"./data/unicycle_inference.pkl")

        file_path = r"./data/unicycle_inference.pkl"
        unicycle_data = load_object(file_path)

        actual_traj = np.vstack(unicycle_data[0])
        ref_traj = np.vstack(unicycle_data[1])
        rdot_traj = np.vstack(unicycle_data[2])

        test_dataset, aug_test_state = make_dataset(N, ref_traj, actual_traj, rdot_traj, rho, horizon)

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
        plt.savefig("./data/inference-rho-v4"+str(rho)+".png")
        #plt.show()

        # Testing new references

        num_inf = 100

        sim_cost, ilqr_cost = test_opt(trained_model_state, 10e-4, aug_test_state, N, num_inf)#, "./data/cost-rho-v2"+str(rho)+"-ilqr.png")

        print(np.mean(sim_cost))
        print(np.mean(ilqr_cost))

        order = 2

        poly_traj, poly_aug_state = polynomial_tests(num_inf, N, order)

        sim_poly_cost, poly_cost = test_opt(trained_model_state, 10e-3, poly_aug_state, N, num_inf)#, "./data/cost-rho-v2"+str(rho)+"-poly.png")

        print(np.mean(sim_poly_cost))
        print(np.mean(poly_cost))

        save_object([sim_cost, ilqr_cost, sim_poly_cost, poly_cost], r"./data/costs-unicycle-rho-v4"+str(rho)+".pkl")


        #plt.title("Testing by opt new references")
        #plt.savefig("./data/test-time.png")
        #plt.show()

        df = pd.DataFrame(np.dstack([sim_poly_cost, poly_cost]).reshape([100, 2]), columns=['opt_sim', 'poly'])

        order=['opt_sim', 'poly']
        x = "Evaluation on Optimizing new references"
        y = "Normalized Relative Tracking Cost"
        flierprops = dict(marker='o', markerfacecolor='#FFFFFF', markersize=4,
                  linestyle='none', markeredgecolor='#D3D3D3')
        plt.figure()
        axes = sns.boxplot(data=pd.melt(df, var_name=x, value_name=y), x=x, y=y, order=order, dodge=False, width=0.5, medianprops=dict(color='black'),
                     # palette={labels[0]:"blue", labels[1]:"orange", labels[2]:"green", labels[3]:"red"}, saturation=1,
                      flierprops=flierprops,
                      showmeans=True, meanprops={"marker":"*", "markerfacecolor":"black", "markeredgecolor":"black"})
        axes.set_xlabel(x, fontsize=15)
        axes.set_ylabel(y, fontsize=15)
        plt.savefig('./data/cost-rho-v4'+str(rho)+'-comp-poly.png', dpi=300, bbox_inches='tight')
        #plt.show()

        df = pd.DataFrame(np.dstack([(np.array(sim_cost)-np.array(ilqr_cost))/np.array(ilqr_cost), (np.array(sim_poly_cost)-np.array(poly_cost))/np.array(poly_cost)]).reshape([100, 2]), columns=['ilqr', 'poly'])

        order=['ilqr', 'poly']
        x = "Evaluation on Optimizing new references"
        y = "Normalized Relative Tracking Cost"
        flierprops = dict(marker='o', markerfacecolor='#FFFFFF', markersize=4,
                  linestyle='none', markeredgecolor='#D3D3D3')
        plt.figure()
        axes = sns.boxplot(data=pd.melt(df, var_name=x, value_name=y), x=x, y=y, order=order, dodge=False, width=0.5, medianprops=dict(color='black'),
                     # palette={labels[0]:"blue", labels[1]:"orange", labels[2]:"green", labels[3]:"red"}, saturation=1,
                      flierprops=flierprops,
                      showmeans=True, meanprops={"marker":"*", "markerfacecolor":"black", "markeredgecolor":"black"})
        axes.set_xlabel(x, fontsize=15)
        axes.set_ylabel(y, fontsize=15)
        plt.savefig('./data/cost-comp-both-rho-v4'+str(rho)+'.png', dpi=300, bbox_inches='tight')
        #plt.show()

        """plt.figure()
        plt.scatter(np.array(sim_cost).shape[0], np.array(sim_cost)/np.array(ilqr_cost))
        plt.plot(np.zeros(100))
        plt.title("Plot of normalized relative cost")
        plt.savefig("./data/rel-cost-ilqr-rho"+str(rho)+".png")
        #plt.show()

        plt.figure()
        plt.scatter(np.array(sim_poly_cost).shape[0], np.array(sim_poly_cost)/np.array(poly_cost))
        plt.plot(np.zeros(100))
        plt.title("Plot of normalized relative cost")
        plt.savefig("./data/rel-cost-poly-rho"+str(rho)+".png")
        #plt.show()"""



if __name__ == '__main__':
    main()






