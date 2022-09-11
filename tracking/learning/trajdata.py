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
    def __init__(self, xtraj, utraj, rtraj, xtraj_, vtraj):
        '''
        Input:
            - xtraj:        np.array(N, p), sequence of states
            - utraj:        np.array(N, q), sequence of inputs
            - rtraj:        np.array(N,), sequence of rewards
            - xtraj_:       np.array(N, p), sequence of next states
            - vtraj:        np.array(N,), the value function for these trajs
        '''
        self.xtraj = torch.tensor(xtraj)
        self.utraj = torch.tensor(utraj)
        self.rtraj = torch.tensor(rtraj)
        self.xtraj_ = torch.tensor(xtraj_)
        self.vtraj = torch.tensor(vtraj)

    def __len__(self):
        return len(self.xtraj)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.xtraj[idx], self.utraj[idx], self.rtraj[idx],\
                self.xtraj_[idx], self.vtraj[idx]

    def cuda(self):
        self.xtraj = self.xtraj.cuda()
        self.utraj = self.utraj.cuda()
        self.rtraj = self.rtraj.cuda()
        self.xtraj_ = self.xtraj_.cuda()
        self.vtraj = self.vtraj.cuda()
