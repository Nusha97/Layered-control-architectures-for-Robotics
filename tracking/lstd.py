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
    Input:      - X:        np.array(N,N), Symmetric matrix X
    Return:     - x:        np.array(N*(N+1)/2), X vectorized
    """
    # Check that X is symmetric
    assert len(X.shape) == 2 and X.shape[0] == X.shape[1]
    assert np.allclose(X, X.T)

    # Vectorize X
    diagonal = np.diag(X)
    triu = X[np.triu_indices_from(X, 1)]
    x = np.concatenate([diagonal, np.sqrt(2) * triu])
    return x


def _smat(x):
    """ The inverse of _svec(). See _svec() above.
    Input:      - x:        np.array(k), vector x
    Return:     - X:        np.array(N,N), x converted into matrix
    """
    # Check that dimension is valid
    assert len(x.shape) == 1
    N = int((np.sqrt(1+8*len(x))-1) / 2)
    assert N*(N+1)/2 == len(x)

    # Convert x into a matrix
    diagonal = np.diag(x[:N])
    triu = np.zeros((N, N))
    triu[np.triu_indices_from(triu, 1)] = x[N:] / np.sqrt(2)
    X = triu + triu.T + diagonal
    return X

def featurize(x, u, K, gamma, sigma=1):
    """ Featurize the state and control action.
    Input:
        - x:        np.array(p), states
        - u:        np.array(q), control actions
        - K:        np.array(q, p), static linear feedback controller
        - gamma:    Float, discount factor
    Output:
        - phi:      np.array(f), the feature phi(x, u) as defined in the
                    paper
    """
    eta = gamma / (1-gamma)
    xu = np.concatenate([x, u])
    IK = np.vstack([np.eye(K.shape[1]), K])
    return _svec(np.outer(xu, xu) + sigma ** 2 * eta * IK @ IK.T)

def lspi(traj, gamma, sigma=1, K0=None, max_iter=100, show_progress=True):
    """ Least squares policy iteration for finding optimal LQR policy
    Input:
        - traj:         List of 4 tuples (x_k, u_k, r_k, x_{k+1})
        - gamma:        discount factor
        - K0:           initial policy
        - max_iter:     maximum number of iterations
    Output:
        - K:            learned controller
    """
    # Initialize controller
    p = len(traj[0][0])
    q = len(traj[0][1])
    if K0 is None:
        K = np.random.random((q, p))
    else:
        K = K0.copy()
    # Policy iteration until convergence
    iter_range = tqdm(range(max_iter)) if show_progress else range(max_iter)
    for _ in iter_range:
        P, _ = evaluate(traj, K, gamma, sigma)
        P12 = P[:p, p:]
        P22 = P[p:, p:]
        K_ = -np.linalg.pinv(P22) @ P12.T
        if np.linalg.norm(K_ - K, 'fro') < PI_TOL:
            break
        K = K_
    # Find the state value function
    IK = np.vstack([np.eye(K.shape[1]), K])
    P_state = IK.T @ P @ IK
    return K, P_state

def evaluate(traj, K, gamma, sigma=1):
    """ Evaluate a given controller K based on the collected data.
    Input:
        - traj:     List of 4 tuples (x_k, u_k, r_k, x_{k+1})
        - K:        np.array(q, p), controller
        - gamma:    Float, discount factor
    Output:
        - Pxu_hat:  State-action value matrix
        - Px_hat:   State value matrix
    """
    # Find the features of points on the trajectory
    phi_xu = np.array([featurize(x, u, K, gamma, sigma) for (x, u, _, _) in traj])
    phi_xx = np.array([featurize(x_, K @ x_, K, gamma, sigma) \
                        for (_, _, _, x_) in traj])
    # Evaluate the policy
    A = np.einsum('ij,ik', phi_xu, (phi_xu-gamma*phi_xx))
    b = np.sum( np.array([r for (_,_,r,_) in traj])[:,None] * phi_xu, 0)
    Pxu_hat = _smat(np.linalg.pinv(A) @ b)
    IK = np.vstack([np.eye(K.shape[1]), K])
    Px_hat = IK.T @ Pxu_hat @ IK
    return Pxu_hat, Px_hat

def construct_traj_list(xtrajs, utrajs, rtrajs):
    """ Convert a list of trajectories into the form that is taken by the
    evaluate method.
    """
    traj = []
    for xtraj, utraj, rtraj in zip(xtrajs, utrajs, rtrajs):
        T = utraj.shape[0]
        traj += list(zip(xtraj[:T], utraj, rtraj, xtraj[1:]))
    return traj

def nominal_to_tracking(A, B, Q, R, T):
    p, q = B.shape
    Z = np.eye(p*T, k=p)
    zero = np.zeros((p, p*T))
    Atilde = np.block([[A, zero],[zero.T, Z]])
    Btilde = np.vstack([B, np.zeros((p*T, q))])
    E = np.hstack([np.eye(p), -np.eye(p), np.zeros((p, p*(T-1)))])
    return Atilde, Btilde, E.T @ Q @ E, R
