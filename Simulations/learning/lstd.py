##############################################################################
# Fengjun Yang, 2022
# Least squares temporal difference learning for low-level tracking control.
#
# This is only applicable to value functions that are linear in features, and
# is thus intended only to be used in testing.
#
# Much of this code has not been vectorized
##############################################################################

PI_TOL = 1e-3

import numpy as np
from tqdm import tqdm


def _svec(X):
    """ Symmetric vectorization: convert a symmetric matrix X compactly into a
    vector x in a way that
                            <X, X> = <x, x>
    Input:      - X:        np.array(n, N,N), Symmetric matrix X
    Return:     - x:        np.array(n, N*(N+1)/2), X vectorized
    """
    diagonal = np.diagonal(X, axis1=1, axis2=2)
    iu = np.triu_indices(X.shape[1], 1)
    triu = X[:, iu[0], iu[1]]
    x = np.hstack([diagonal, np.sqrt(2) * triu])
    return x

def _smat(x):
    """ The inverse of _svec(). See _svec() above.
    Input:      - x:        np.array(n, k), vector x
    Return:     - X:        np.array(n, N,N), x converted into matrix
    """
    N = int((np.sqrt(1+8*x.shape[1])-1) / 2)
    assert N*(N+1)/2 == x.shape[1]
    # Convert the stacked diagonal elements into stacked diagonal matrices
    diagonal = x[:, None, :N] * np.eye(N)
    iu = np.triu_indices(N, 1)
    triu = np.zeros((x.shape[0], N, N))
    triu[:, iu[0], iu[1]] = x[:, N:] / np.sqrt(2)
    X = diagonal + triu + np.transpose(triu, (0,2,1))
    return X

def featurize(xs, us, K, gamma, sigma=0):
    """ Featurize the state and control action.
    Input:
        - xs:       np.array(n, p), states
        - us:       np.array(n, q), control actions
        - K:        np.array(q, p), static linear feedback controller
        - gamma:    Float, discount factor
    Output:
        - phi:      np.array(n, f), the feature phi(x, u) as defined in the
                    paper
    """
    eta = gamma / (1-gamma)
    xus = np.hstack([xs, us])
    xu_outer = xus[:, :, None] * xus[:, None, :]
    IK = np.vstack([np.eye(K.shape[1]), K])
    IKIK = sigma ** 2 * eta * IK @ IK.T
    return _svec(xu_outer + IKIK)

def evaluate(xtraj, utraj, rtraj, xtraj_, K, gamma, sigma=0):
    """ Evaluate a given controller K based on the collected data.
    Input:
        - xtraj:    np.array(n, p), trajectory of state
        - utraj:    np.array(n, q), trajectory of input
        - rtraj:    np.array(n, 1), trajectory of reward
        - xtraj_:   np.array(n, p), trajectory of next state
        - K:        np.array(q, p), controller
        - gamma:    Float, discount factor
    Output:
        - Pxu_hat:  State-action value matrix
        - Px_hat:   State value matrix
    """
    # Find the features of points on the trajectory
    phi_xu = featurize(xtraj, utraj, K, gamma, sigma)
    utraj_ = K @ xtraj_[:,:,None]
    phi_xx = featurize(xtraj_, utraj_[:, :, 0], K, gamma, sigma)
    # Evaluate the policy
    A = np.einsum('ij,ik', phi_xu, (phi_xu-gamma*phi_xx))
    b = np.sum( rtraj * phi_xu, 0)
    pxu_hat = np.linalg.pinv(A) @ b
    Pxu_hat = _smat(pxu_hat[None, :])[0]
    IK = np.vstack([np.eye(K.shape[1]), K])
    Px_hat = IK.T @ Pxu_hat @ IK
    return Pxu_hat, Px_hat

def lspi(xtraj, utraj, rtraj, xtraj_, gamma, sigma=0, K0=None,
         max_iter=50, verbose=False):
    """ Least squares policy iteration for finding optimal LQR policy
    Input:
        - xtraj:    np.array(n, p), trajectory of state
        - utraj:    np.array(n, q), trajectory of input
        - rtraj:    np.array(n, 1), trajectory of reward
        - xtraj_:   np.array(n, p), trajectory of next state
        - gamma:        discount factor
        - K0:           initial policy
        - max_iter:     maximum number of iterations
    Output:
        - K:            learned controller
    """
    # Initialize controller
    p = xtraj.shape[1]
    q = utraj.shape[1]
    if K0 is None:
        K = np.random.random((q, p))
    else:
        K = K0.copy()
    # Policy iteration until convergence
    for it in range(max_iter):
        P, _ = evaluate(xtraj, utraj, rtraj, xtraj_, K, gamma, sigma)
        P12 = P[:p, p:]
        P22 = P[p:, p:]
        K_ = -np.linalg.pinv(P22) @ P12.T
        improvement = np.linalg.norm(K_ - K, 'fro')
        if verbose:
            print('Iteration:{:6d}, |P - P_new|:\t{:8.3f}'.format(it, improvement))
        if improvement < PI_TOL:
            break
        K = K_
    # Find the state value function
    IK = np.vstack([np.eye(K.shape[1]), K])
    P_state = IK.T @ P @ IK
    return K, P_state
