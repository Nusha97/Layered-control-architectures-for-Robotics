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
