import numpy as np
from abc import ABC, abstractmethod

class BaseController:
    ''' Base for all controllers '''
    @abstractmethod
    def control(self, y):
        pass

    @abstractmethod
    def reset(self):
        pass

class RandomController(BaseController):
    ''' Outputs white-noise as control inputs '''
    def __init__(self, q):
        self.q = q

    def control(self, y):
        ''' Outputs random control
        Input:
            - y:        np.array(n, r), batched states
        '''
        return np.random.randn(y.shape[0], self.q)

    def reset(self):
        # Nothing to reset here
        pass

class ZeroController(BaseController):
    ''' Outputs zero control inputs '''
    def __init__(self, q):
        self.q = q

    def control(self, y):
        ''' Outputs zeros
        Input:
            - y:        np.array(n, r), batched states
        '''
        return np.random.zeros((y.shape[0], self.q))

    def reset(self):
        # Nothing to reset here
        pass

class LinearFbController(BaseController):
    def __init__(self, K):
        '''
        Input:
            - K:        np.array(q, r), feedback gain from observation
        '''
        self.K = K
        self.counter = 0
        if isinstance(K, list):
            self.dynamic = True
        else:
            self.dynamic = False

    def control(self, y):
        ''' Outputs zeros
        Input:
            - y:        np.array(n, r), batched states
        '''
        if self.dynamic:
            K = self.K[self.counter]
        else:
            K = self.K
        output = (K @ y.T).T
        self.counter += 1
        return output

    def reset(self):
        self.counter = 0
