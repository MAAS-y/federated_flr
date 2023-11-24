"""Tucker tensor times matrix forward and backward passes"""
import numpy as np
import torch

#Author: Zi Yang

def tucker_times_matrix_fwd(tensor, matrix):
    """Tucker tensor times matrix forward pass"""
    #you must replace the internals here
    # output = test_tensor_times_matrix(tensor,matrix)
    with torch.no_grad():
        core = tensor.core

        N = int(len(tensor.factors) / 2)

        size = [x.shape[0] for x in tensor.factors]

        out_shape = [matrix.shape[1], np.prod(size[N:])]

        output = (matrix.T).reshape([matrix.shape[1]] + size[:N])

        for i in range(N):
            output = torch.tensordot(output,
                                     tensor.factors[i],
                                     dims=[[1], [0]])

        output = torch.tensordot(output,
                                 tensor.core,
                                 dims=[list(range(1, N + 1)),
                                       list(range(N))])

        for i in range(N):
            output = torch.tensordot(output,
                                     tensor.factors[i + N],
                                     dims=[[1], [1]])
        output = output.reshape(out_shape)

        saved_tensors = [matrix]
    return output, saved_tensors


def tucker_times_matrix_bwd(tensor, dy, saved_tensors):
    """
    Tucker tensor times matrix backward pass
    """
    #the example below is just a dummy, you have to compute your own gradients
    with torch.no_grad():

        matrix = saved_tensors[0]
        matrix_shape = matrix.shape

        dtype = tensor.core.dtype
        device = tensor.core.device

        N = int(len(tensor.factors) / 2)

        size = [x.shape[0] for x in tensor.factors]

        # dymatrix_tensor = (matrix @ dy).reshape(size)

        matrix = (matrix.T).reshape([matrix.shape[1]] + size[:N])
        matrix_contr = matrix.clone()

        for A in tensor.factors[:N]:
            matrix_contr = torch.tensordot(matrix_contr, A, dims=[[1], [0]])

        dy = dy.reshape([dy.shape[0]] + size[N:])
        dy_contr = dy.clone()
        for A in tensor.factors[N:]:
            dy_contr = torch.tensordot(dy_contr, A, dims=[[1], [0]])

        grad_core = torch.tensordot(matrix_contr, dy_contr, dims=[[0], [0]])

        matrix_contr = torch.tensordot(
            matrix_contr,
            tensor.core,
            dims=[list(range(1, N + 1)), list(range(N))])
        dy_contr = torch.tensordot(
            dy_contr,
            tensor.core,
            dims=[list(range(1, N + 1)),
                  list(range(N, 2 * N))])

        grads = []

        for j in range(N):
            grad_j = matrix.clone()
            for A in tensor.factors[0:j]:
                grad_j = torch.tensordot(grad_j, A, dims=[[1], [0]])
            for A in tensor.factors[j + 1:N]:
                grad_j = torch.tensordot(grad_j, A, dims=[[2], [0]])
            grad_j = torch.tensordot(grad_j,
                                     dy_contr,
                                     dims=[[0] + list(range(2, N + 1)),
                                           list(range(0, j + 1)) +
                                           list(range(j + 2, N + 1))])
            grads.append(grad_j)

        for j in range(N, 2 * N):
            grad_j = dy.clone()
            for A in tensor.factors[N:j]:
                grad_j = torch.tensordot(grad_j, A, dims=[[1], [0]])
            for A in tensor.factors[j + 1:]:
                grad_j = torch.tensordot(grad_j, A, dims=[[2], [0]])
            grad_j = torch.tensordot(grad_j,
                                     matrix_contr,
                                     dims=[[0] + list(range(2, N + 1)),
                                           list(range(0, j + 1 - N)) +
                                           list(range(j + 2 - N, N + 1))])
            grads.append(grad_j)

        dx = dy.clone()

        for A in tensor.factors[N:]:
            dx = torch.tensordot(dx, A, dims=[[1], [0]])
        dx = torch.tensordot(
            dx,
            tensor.core,
            dims=[list(range(1, N + 1)),
                  list(range(N, 2 * N))])

        for A in tensor.factors[0:N]:
            dx = torch.tensordot(dx, A, dims=[[1], [1]])

        dx = dx.reshape([matrix_shape[1], matrix_shape[0]]).T

    grads.append(grad_core)
    return grads, dx


###########################################
# Only modify this block
# You replace these functions with your own
# This is just a dummy
###########################################
