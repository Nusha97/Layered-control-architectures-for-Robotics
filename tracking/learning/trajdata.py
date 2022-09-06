##############################################################################
# Fengjun Yang, 2022
# This file contains the definition of the trajectory dataset class and helper
# functions that aid in generating these classes.
##############################################################################

import torch
from torch.utils.data import Dataset, DataLoader

class TrajDataset(Dataset):
    ''' Trajectory dataset; inherits torch.utils.data.Dataset for easier
        training. The data are transformed into torch tensors
    '''
    def __init__(self, xtraj, utraj, rtraj, xtraj_):
        '''
        Input:
            - xtraj:        np.array(N, p), sequence of states
            - utraj:        np.array(N, q), sequence of inputs
            - rtraj:        np.array(N,), sequence of rewards
            - xtraj_:       np.array(N, p), sequence of next states
        '''
        self.xtraj, self.utraj, self.rtraj, self.xtraj_ = \
                xtraj, utraj, rtraj, xtraj_

    def __len__(self):
        return len(self.xtraj)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.xtraj[idx]), torch.tensor(self.utraj[idx]), \
                torch.tensor(self.rtraj[idx]), torch.tensor(self.xtraj_[idx])

    def cuda(self):
        self.xtraj = self.xtraj.cuda()
        self.utraj = self.utraj.cuda()
        self.rtraj = self.rtraj.cuda()
        self.xtraj_ = self.xtraj_.cuda()
