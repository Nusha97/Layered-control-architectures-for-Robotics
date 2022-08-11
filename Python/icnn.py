import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class ICNN(nn.Module):
    """ Input Convex Neural Network
    """
    def __init__(self, input_size, widths):
        """ Constructor
        Parameters:
            - widths:       list of Integers indicating the width of each layer
        """
        super().__init__()
        self.Wys = [torch.nn.Parameter(torch.randn((w, input_size), dtype=torch.double)) \
                for w in widths]
        self.Wzs = [torch.nn.Parameter(torch.randn((widths[i+1], widths[i]), dtype=torch.double)) for i in range(len(widths)-1)]
        self.bs = [torch.nn.Parameter(torch.randn(w, dtype=torch.double)) \
                   for w in widths]
        self.activate = nn.ELU()
        for i, W in enumerate(self.Wys):
            self.register_parameter(name='Wy'+str(i), param=W)
        for i, W in enumerate(self.Wzs):
            self.register_parameter(name='Wz'+str(i+1), param=W)
        for i, b in enumerate(self.bs):
            self.register_parameter(name='b'+str(i), param=b)

    def forward(self, y):
        # First layer
        z = self.activate(torch.matmul(self.Wys[0], y).T + self.bs[0]).T
        # Hidden layers
        for Wy, Wz, b in zip(self.Wys[1:-1], self.Wzs[:-1], self.bs[1:-1]):
            z = torch.matmul(torch.exp(Wz), z).T + torch.matmul(Wy, y).T + b
            z = self.activate(z).T
        # Last layer
        z = torch.matmul(torch.exp(self.Wzs[-1]), z).T + torch.matmul(self.Wys[-1], y).T + self.bs[-1]
        return z
