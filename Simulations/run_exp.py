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
from generate_data import gen_uni_training_data, ILQR, unicycle, save_object, unicycle_K
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


Kp = 5 * np.array([[2, 1, 0], [0, 1, 3]])
gamma = 0.99

# Generate training data
def data_generation(num_iter, file_path):
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
def make_dataset(N, ref_traj, actual_traj, rdot_traj, horizon=101):
    q = 2
    p = 3 + 3 * N
    traj_len = ref_traj.shape[0]
    num_iter = int(traj_len / horizon)

    cost_traj, input_traj = compute_tracking_cost(ref_traj, actual_traj, rdot_traj, Kp, N, horizon)

    aug_state = []
    for i in range(num_iter):
        r0 = ref_traj[i * horizon:(i + 1) * horizon, :]
        r0 = np.append(r0, r0[-1, :] * np.ones((N - 1, 3)))
        r0 = np.reshape(r0, (horizon + N - 1, 3))
        for j in range(horizon - N):
            aug_state.append(np.append(actual_traj[j, :], r0[j:j + N, :]))
    # aug_state = [np.append(actual_traj[r, :], ref_traj[r:r+N, :]) for r in range(num_iter)]
    aug_state = np.array(aug_state)
    print(aug_state.shape)

    Tstart = 0
    Tend = aug_state.shape[0]

    dataset = TrajDataset(aug_state[Tstart:Tend - 1, :].astype('float64'),
                          input_traj[Tstart:Tend - 1, :].astype('float64'),
                          cost_traj[Tstart:Tend - 1, None].astype('float64'),
                          aug_state[Tstart + 1:Tend, :].astype('float64'))

    return dataset


def load_model():
    with open(r"/home/anusha/Research/Layered-architecture-quadrotor-control/Simulations/data/params.yaml") as f:
        yaml_data = yaml.load(f, Loader=yaml.RoundTripLoader)


    num_hidden = yaml_data['num_hidden']
    batch_size = yaml_data['batch_size']
    learning_rate = yaml_data['learning_rate']
    num_epochs = yaml_data['num_epochs']
    model_save = yaml_data['save_path']

    model = MLP(num_hidden=num_hidden, num_outputs=1)

    rng = jax.random.PRNGKey(427)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (batch_size, p))  # Batch size 64, input size p
    # Initialize the model
    params = model.init(init_rng, inp)

    optimizer = optax.adam(learning_rate=learning_rate)

    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=optimizer)

    return model, batch_size, model_state, num_epochs



def main():

    num_iter = 10000
    file_path = r"/home/anusha/Research/Layered-architecture-quadrotor-control/Simulations/data/uni_train-nonoise3.pkl"
    ref_traj, actual_traj, rdot_traj = data_generation(num_iter, file_path)

    train_dataset = make_dataset(50, ref_traj, actual_traj, rdot_traj, 101)

    model, batch_size, model_state, num_epochs = load_model()

    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
    trained_model_state = train_model(model_state, train_data_loader, num_epochs=num_epochs)

    eval_model(trained_model_state, train_data_loader, batch_size)

    trained_model = model.bind(trained_model_state.params)

    # Inference
    ref_traj, actual_traj, rdot_traj = data_generation(num_iter=1000, file_path=)

    test_dataset = make_dataset(5, ref_traj, actual_traj, rdot_traj, 101)

    test_data_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)
    eval_model(trained_model_state, test_data_loader, batch_size)

    # Save plot on entire test dataset
    out = []
    true = []
    for batch in test_data_loader:
        data_input, _, cost, _ = batch
        out.append(trained_model(data_input))
        true.append(cost)



    # Testing new references






