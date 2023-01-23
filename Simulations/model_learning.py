"""
SYNOPSIS
    Contains functions required to load data, train and run inference on
    neural network models for learning value function from trajectories.
DESCRIPTION

    Implementation uses JAX libraries (flax) to define functions to train,
    evaluate and save deep learning models.
AUTHOR

    Anusha Srikanthan <sanusha@seas.upenn.edu>
LICENSE

VERSION
    0.0
"""
from flax import linen as nn
from flax.training import checkpoints # need to install tensorflow
import torch.utils.data as data
import numpy as np
import optax

## Progress bar
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from torch.utils.data import Dataset
import torch
import cvxpy as cp
from jax.scipy.optimize import minimize


class TrajDataset(Dataset):
    """
    Dataset class inherited from torch modules
    """
    def __init__(self, xtraj, utraj, rtraj, xtraj_):
        """
        Input:
            - xtraj:        np.array(N, p), sequence of states
            - utraj:        np.array(N, q), sequence of inputs
            - rtraj:        np.array(N,), sequence of rewards
            - xtraj_:       np.array(N, p), sequence of next states
        """
        self.xtraj = xtraj
        self.utraj = utraj
        self.rtraj = rtraj
        self.xtraj_ = xtraj_

    def __len__(self):
        return len(self.xtraj)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.xtraj[idx], self.utraj[idx], self.rtraj[idx], self.xtraj_[idx]


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def calculate_loss(state, params, batch):
    data_state, data_input, data_cost, data_next = batch
    pred = state.apply_fn(params, data_state)
    target = data_cost

    # Calculate the loss
    loss = optax.l2_loss(pred.ravel(), target.ravel()).mean()
    return loss


@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(calculate_loss,  # Function to calculate the loss
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=False  # Function has additional outputs, here accuracy
                                )
    # Determine gradients for current model, parameters and batch
    loss, grads = grad_fn(state, state.params, batch)
    print(loss)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss


@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    # Determine the accuracy
    loss = calculate_loss(state, state.params, batch)
    return loss


def train_model(state, data_loader, num_epochs=100):
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss = train_step(state, batch)
            # We could use the loss and accuracy for logging here, e.g. in TensorBoard
            # For simplicity, we skip this part here
    return state


def eval_model(state, data_loader, batch_size):
    all_losses, batch_sizes = [], []
    for batch in data_loader:
        batch_loss = eval_step(state, batch)
        all_losses.append(batch_loss)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    loss = sum([a*b for a, b in zip(all_losses, batch_sizes)]) / sum(batch_sizes)
    print(f"Loss of the model: {loss:4.2f}")


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir, step=0):
    checkpoints.save_checkpoint(workdir, state, step)


def inference_model(state, data_loader):
    """
    At inference time, we evaluate the model on the holdout dataset
    :return:
    """
    eval_model(state, data_loader, batch_size)


def calculate_cost(data_state, init, goal, state, params):
    pred = state.apply_fn(params, jnp.append(init, data_state)).ravel()
    return pred[0] + jnp.linalg.norm(data_state[0:3] - init) ** 2#+ jnp.linalg.norm(data_state[-3:] - goal) ** 2  # Adding terminal state constraint to the objective function


def test_model(trained_state, data_loader, batch_size):
    """
    At test time, we optimize for a reference trajectory that minimizes LQR cost + regularizer
    Need to add penalty for regularizer and change cost
    :return:
    """
    data_state, _, data_cost, _ = next(iter(data_loader))
    solution = []
    orig = []
    for data, cost in zip(data_state, data_cost):
        # print("Cost computed using the network", calculate_cost(data, trained_state, trained_state.params))
        # print("True cost", cost)
        solution.append(minimize(calculate_cost, data[3:], args=(data[0:3], data[-3:], trained_state, trained_state.params), method="BFGS"))
        orig.append(data)
    # Return new reference and cost
    return solution, orig




