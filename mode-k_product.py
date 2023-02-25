import numpy as np
import tensorly as tl
import torch
from time import time

# # tensor mode k unfolding, mode k unfolding is the way you read the entry of a tensor
# X is tensor; U is matrix


## -----------------------------#
## method 1  definition
## -----------------------------#

# X = [[[a0 + d * 3, a0 + d * 3 + 12] for d in range(0, 4)] for a0 in range(1, 4)]
# X = np.array(X)

# fibers_mode1 = [X[:, j, k] for k in range(0, 2) for j in range(0, 4)]
# fibers_mode2 = [X[i, :, k] for k in range(0, 2) for i in range(0, 3)]
# fibers_mode3 = [X[i, j, :] for j in range(0, 4) for i in range(0, 3)]
# unfold_mode1 = np.array(fibers_mode1).T
# unfold_mode2 = np.array(fibers_mode2).T
# unfold_mode3 = np.array(fibers_mode3).T


## -----------------------------#
## method 2  swapaxes (unable)
## -----------------------------#
# def mode_n_product(U, X, mode):
#    X = np.asarray(X)
#    U = np.asarray(U)
#    if mode <= 0 or mode % 1 != 0:
#        raise ValueError('`mode` must be a positive interger')
#    if X.ndim < mode:
#        raise ValueError('Invalid shape of X for mode = {}: {}'.format(mode, x.shape))
#    if U.ndim != 2:
#        raise ValueError('Invalid shape of M: {}'.format(m.shape))
#    return np.swapaxes(np.swapaxes(X, mode - 1, -1).dot(U.T), mode - 1, -1)


# # -----------------------------#
# # method 3  transpose
# # -----------------------------#
# def mode_k_product(U, X, mode):
#     axes1 = list(range(X.ndim))

#     axes1[mode] = 0
#     axes1[0] = mode
#     Y = np.transpose(X, axes1)
#     axes2 = list(Y.shape)
#     Y = np.reshape(Y, (Y.shape[0], -1))
#     Y = U @ Y
#     axes2[0] = Y.shape[0]
#     Y = np.reshape(Y, axes2)
#     Y = np.transpose(Y, axes1)
#     return Y


# # -----------------------------#
# # method 4  numpy einsum_product
# # -----------------------------#
# def einsum_numpy(U, X, mode):
#     axes1 = list(range(X.ndim))
#     axes1[mode] = X.ndim + 1
#     axes2 = list(range(X.ndim))
#     axes2[mode] = X.ndim
#     return np.einsum(U, [X.ndim, X.ndim + 1], X, axes1, axes2, optimize=True)


# -----------------------------#
# method 5  pytorch einsum_product
# -----------------------------#
def einsum_torch(U, X, args):  # args is string
    return torch.einsum("pqr,sp->sqr", [X, U])


# -----------------------------#
# method 5  tensorly
# -----------------------------#
def tensorly_mode_dot(U, X, mode):  
    # different from other methods, tensorly mode begin from 0, not 1.
    return tl.tenalg.mode_dot(X, U, mode=0)  # mode 0 => 1mode product


# torch_org_out = torch.tensordot(a, b, dims=([1, 2], [2, 4])).numpy()


# def test_correctness():
#     A = np.random.rand(3, 4, 5)
#     for i in range(3):
#         B = np.random.rand(6, A.shape[i])
#         X = mode_k_product(B, A, i)
#         Y = einsum_numpy(B, A, i)
#         print(np.allclose(X, Y))


# def test_time(method, amount):
#     U = np.random.rand(256, 512)
#     X = np.random.rand(512, 512, 256)
#     start = time()
#     for i in range(amount):
#         method(U, X, 1)
#     return (time() - start) / amount


# def test_times():
#     print("Transpose:", test_time(mode_k_product, 10))
#     print("Einsum:", test_time(einsum_product, 10))
#     # print("swapaxes:", test_time(ultra_mode_k_product, 0))


# test_correctness()
# test_times()
