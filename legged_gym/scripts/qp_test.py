import torch
import torch.multiprocessing as mp
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import numpy as np
import scipy
from time import time

def random_batch_qp(batch_size, n):
    M, b = np.random.random((n, n)), np.random.random(n)
    P, q = np.dot(M.T, M), np.dot(b, M).reshape((n,))
    G = scipy.linalg.toeplitz([1., 0., 0.] + [0.] * (n - 3), [1., 2., 3.] + [0.] * (n - 3))
    h = np.ones(n)
    P_sqrt = np.sqrt(P)
    # Copy to make batch
    P_sqrt_batch = torch.tensor(np.repeat(np.expand_dims(P_sqrt, 0), batch_size, axis=0))
    q_batch = torch.tensor(np.repeat(np.expand_dims(q, 0), batch_size, axis=0))
    G_batch = torch.tensor(np.repeat(np.expand_dims(G, 0), batch_size, axis=0))
    h_batch = torch.tensor(np.repeat(np.expand_dims(h, 0), batch_size, axis=0))
    return (P_sqrt_batch, q_batch, G_batch, h_batch)

def build_qp_layer(n):

      # Define and solve the CVXPY problem.
      P_sqrt = cp.Parameter((n, n))
      q = cp.Parameter((n))
      G = cp.Parameter((n, n))
      h = cp.Parameter((n))
      x = cp.Variable(n)
      prob = cp.Problem(cp.Minimize(0.5*cp.sum_squares(P_sqrt @ x) + q.T @ x), [G @ x <= h])
      assert prob.is_dpp()
      qp_layer = CvxpyLayer(prob, parameters=[P_sqrt, q, G, h], variables=[x])
      return qp_layer

if __name__ == "__main__":

    batch_size = 4096
    n = 6

    diff_qp_layer = build_qp_layer(n)
    P_sqrt_batch, q_batch, G_batch, h_batch = random_batch_qp(batch_size, n)

    # Solve without multi-processing
    start_time = time()
    result = diff_qp_layer(P_sqrt_batch, q_batch, G_batch, h_batch)[0]
    print('solution_time = {}'.format(time() - start_time))
    # solution_time ~ 1.6525659561157227 for batch_size 512 n = 5

    # Solve with multi-processing
    start_time = time()
    n_jobs = 2
    pool = mp.Pool(n_jobs)
    args = []
    batch_size_m = int(batch_size / n_jobs) + 1
    for i in range(n_jobs):
        i_str = i*batch_size_m
        i_end = min((i+1)*batch_size_m, batch_size)
        args.append((P_sqrt_batch[i_str:i_end], q_batch[i_str:i_end], G_batch[i_str:i_end], h_batch[i_str:i_end]))
    results = pool.starmap(diff_qp_layer, args)
    print('solution_time with mp = {}'.format(time() - start_time))
    # solution_time with mp ~ 0.9894797801971436 for batch_size = 512 n = 5 n_jobs = 2
    # solution_time with mp ~ 0.6761658191680908 for batch_size = 512 n = 5 n_jobs = 3
    # solution_time with mp ~ 0.9608180522918701 for batch_size = 512 n = 5 n_jobs = 4
