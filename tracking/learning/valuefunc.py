################################################################################
# Fengjun Yang, 2022
# This file contains the various value function parameterizations.
################################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from . import lstd

class ValueFunc(ABC):
    ''' This is the abstract class of value functions. It is inherited by
        different specific parameterizations.
    '''
    @abstractmethod
    def learn(self, dataset):
        pass

    @abstractmethod
    def pred(self, x0, ref):
        pass

################################################################################
# Least squares parameterization
################################################################################
class QuadraticValueFunc(ValueFunc):
    ''' Parameterizes the value function as a quadratic function and learn it
        with least squares temporal difference. Does not backprop.
    '''
    def __init__(self, gamma=0.99, sigma=0):
        self.Pxu = None
        self.P = None
        self.gamma = gamma
        self.sigma = sigma

    def learn(self, dataset, K):
        xtraj, utraj, rtraj, xtraj_ = dataset[:]
        self.Pxu, self.P = lstd.evaluate(xtraj.numpy(), utraj.numpy(),
                rtraj.numpy(), xtraj_.numpy(), K, self.gamma, self.sigma)

    def pred(self, x0, ref):
        d0 = np.concatenate([x0, ref])
        return np.dot(d0, self.P @ d0)


################################################################################
# Neural network parameterizations
################################################################################

class NNValueFunc(ValueFunc):
    ''' This class parameterizes the value function as a neural network
    '''
    def __init__(self):
        ''' Initialize a neural network '''
        self.network = None

    def learn(self, dataset, gamma, num_epoch=100, lr=0.01, batch_size=64,
              verbose=False, print_interval=10, beta=1):
        ''' The general training loop for NN value functions '''
        # Define loss function, optimizer and lr scheduler
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        lossfn = nn.HuberLoss()
        optimizer = optim.Adam(self.network.parameters(), lr=lr,
                weight_decay=1e-2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2,
                gamma=0.99)
        # Training loop
        for epoch in range(num_epoch):
            for x, _, r, x_, v in dataloader:
                self.network.zero_grad()
                # Compute loss as the difference of prediction and target
                pred = self.network(x)
                target = r + gamma * self.network(x_)
                loss = lossfn(pred, v) + beta * lossfn(pred, target)
                # Update weights and learning rate
                loss.backward()
                optimizer.step()
                #scheduler.step()
            #beta = beta * 1.005
            # Print out the loss if in verbose mode
            if(verbose and epoch % print_interval == 0):
                print('Epoch: {} \t Training loss: {}'.format(epoch+1, loss.item()))

    def eval(self):
        for param in self.network.parameters():
            param.requires_grad = False
        self.network.eval()

    def pred(self, x0, ref):
        ''' The general prediction for NN value functions '''
        d0 = torch.cat([x0, ref]).double()
        return self.network(d0.unsqueeze(0))[0]

    def cuda(self):
        self.network.cuda()

    def cpu(self):
        self.network.cpu()

class ICNNValueFunc(NNValueFunc):
    ''' Value functions parameterized as a convex neural network '''
    def __init__(self, input_size, widths):
        ''' Constructor
        Parameters:
            - widths:       list of Integers indicating the width of each layer
        '''
        self.network = ICNN(input_size, widths)


class MLPValueFunc(NNValueFunc):
    ''' Value functions paramterized as a multi-layer-perceptron '''
    def __init__(self, input_size, widths):
        ''' Constructor
        Parameters:
            - widths:       list of Integers indicating the width of each layer
        '''
        self.network = MLP(input_size, widths)


################################################################################
# The various neural networks that we use to parameterize the value functions
################################################################################

class ICNN(nn.Module):
    """ Input Convex Neural Network
    """
    def __init__(self, input_size, widths):
        """ Constructor
        Parameters:
            - widths:       list of Integers indicating the width of each layer
        """
        super().__init__()
        self.Wys = [torch.nn.Parameter(torch.randn((1, w, input_size), dtype=torch.double)) \
                for w in widths]
        self.Wzs = [torch.nn.Parameter(
                        torch.randn((1, widths[i+1], widths[i]), dtype=torch.double)) \
                                    for i in range(len(widths)-1)]
        self.bs = [torch.nn.Parameter(torch.randn(1, w, 1, dtype=torch.double)) \
                   for w in widths]
        self.activate = nn.ELU()
        for i, W in enumerate(self.Wys):
            self.register_parameter(name='Wy'+str(i), param=W)
        for i, W in enumerate(self.Wzs):
            self.register_parameter(name='Wz'+str(i+1), param=W)
        for i, b in enumerate(self.bs):
            self.register_parameter(name='b'+str(i), param=b)

    def forward(self, y):
        y_ = y.unsqueeze(-1)
        # First layer
        z = self.activate(self.Wys[0] @ y_ + self.bs[0])
        # Hidden layers
        for Wy, Wz, b in zip(self.Wys[1:-1], self.Wzs[:-1], self.bs[1:-1]):
            z = torch.exp(Wz) @ z + Wy @ y_ + b
            z = self.activate(z)
        # Last layer
        z = torch.exp(self.Wzs[-1]) @ z + self.Wys[-1] @ y_ + self.bs[-1]
        return torch.squeeze(z, 1)

class MLP(nn.Module):
    ''' Multi-layer perceptron '''
    def __init__(self, input_size, widths):
        ''' Constructor '''
        super().__init__()
        # First layer
        layers = [nn.Linear(input_size, widths[0]).double(), nn.ReLU()]
        # Hidden layers
        for w1, w2 in zip(widths[0:-2], widths[1:-1]):
            layers += [nn.Linear(w1, w2).double(), nn.ReLU()]
        # Last layer
        layers.append(nn.Linear(widths[-2], widths[-1]).double())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
