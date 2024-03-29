"""
SYNOPSIS
    Contains functions required to load data, train and run inference on
    neural network models for learning value function from trajectories.

DESCRIPTION
    Implementation uses JAX libraries (flax) to define functions to train,
    evaluate and save deep learning models.

AUTHOR
    Anusha Srikanthan <sanusha@seas.upenn.edu>

VERSION
    0.0
"""

from flax.training import checkpoints  # need to install tensorflow
import numpy as np
import optax

from tqdm.auto import tqdm

import jax
from torch.utils.data import Dataset
import torch


class TrajDataset(Dataset):
    """
    Dataset class inherited from torch modules
    """
    def __init__(self, xtraj, utraj, rtraj, xtraj_):
        """
        Input:
            - xtraj:        np.array(N, p), sequence of states
            - utraj:        np.array(N, q), sequence of inputs
            - rtraj:        np.array(N,), sequence of costs
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
    """
    A numpy helper function for efficient batching from JAX documentation
    :param batch: batches from the dataset
    :return: batch samples
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def calculate_loss(state, params, batch):
    """
    Loss function for training defined as the l2 loss between prediction and target
    :param state:
    :param params:
    :param batch:
    :return:
    """
    data_state, data_input, data_cost, data_next = batch
    pred = state.apply_fn(params, data_state)
    target = data_cost

    # Calculate the loss
    loss = optax.l2_loss(pred.ravel(), target.ravel()).mean()

    return loss


@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    """
    One iteration of training by running the back propagation through the batch
    :param state: weights of the neural network model
    :param batch: batch from the dataset
    :return: state, loss
    """
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
    """
    One iteration of predictions to evaluate the model
    :param state: weights of the neural network model
    :param batch: batch from the dataset
    :return: loss
    """
    # Determine the accuracy
    loss = calculate_loss(state, state.params, batch)
    return loss


def train_model(state, data_loader, num_epochs=100):
    """
    Train the model over the training dataset
    :param state: weights of the neural network model
    :param data_loader: batched dataset
    :param num_epochs: number of epochs
    :return: state
    """
    # Training loop
    count = 0
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for batch in data_loader:
            count += 1
            state, loss = train_step(state, batch)
            epoch_loss += loss
    return state


def eval_model(state, data_loader, batch_size):
    """
    Evaluate model over the test dataset
    :param state: weights of the neural network model
    :param data_loader: batched dataset
    :param batch_size: number of samples in a batch
    :return: None
    """
    all_losses, batch_sizes = [], []
    for batch in data_loader:
        batch_loss = eval_step(state, batch)
        all_losses.append(batch_loss)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    loss = sum([a*b for a, b in zip(all_losses, batch_sizes)]) / sum(batch_sizes)
    print(f"Loss of the model: {loss:4.2f}")


def restore_checkpoint(state, workdir):
    """
    Restore the weights of the model
    :param state: initialized model object
    :param workdir: file path
    :return: state of the network
    """
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir, step=0):
    """
    Save the weights to a file
    :param state: model object
    :param workdir: file path
    :param step: checkpoint index
    :return: None
    """
    checkpoints.save_checkpoint(workdir, state, step)






