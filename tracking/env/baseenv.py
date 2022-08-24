###############################################################################
# Fengjun Yang, 2022
# Base environment, will be inherited by different instantiations.
###############################################################################

from abc import abstractmethod
from . import controller

class BaseEnv:
    ''' The abstract environment class
    '''
    def __init__(self, p, q, r):
        '''
        Input:
            - p:        Integer, state dimension
            - q:        Integer, control dimension
            - r:        Integer, observation dimension
        '''
        # System dimensions
        self.p = p
        self.q = q
        self.r = r
        # System functions
        self.reward = None
        self.observe = None

    @abstractmethod
    def step(self, x, u):
        ''' Evolve x and u to the next state '''
        pass

    @abstractmethod
    def observe(self, x):
        ''' Apply observation noise to state '''
        pass

    @abstractmethod
    def reset(self):
        ''' Reset variables that are state-dependent '''
        pass

    @abstractmethod
    def totracking(self):
        pass

    @abstractmethod
    def sampletraj(self, x0s=None):
        pass

    def randctrl(self):
        return controller.RandomController(self.q)

    def zeroctrl(self):
        return controller.ZeroController(self.q)
