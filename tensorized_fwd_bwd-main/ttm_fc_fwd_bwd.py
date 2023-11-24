"""
TTM times matrix forward and backward
"""
import itertools
from math import prod
import torch


#Author Angela Chen

def ttm_times_matrix_fwd(tensor, matrix):
    """TTM times matrix forward

    Parameters
    ----------
    tensor : BlockTT
        TTM tensorized weight matrix
    matrix : Parameter object
        input matrix x

    Returns
    -------
    output
        tensor times matrix
        equivalent to matrix.T@tensor.to_matrix()
    saved_tensors
        tensorized input matrix
    """

    # Prepare tensor shape
    shape_x = matrix.T.shape
    tensorized_shape_x, tensorized_shape_y = tensor.tensorized_shape

    num_batch = shape_x[0]
    order = len(tensor.factors)

    # Reshape transpose of input matrix to input tensor
    input_tensor = torch.reshape(matrix.T, (shape_x[0], ) + tensorized_shape_x)

    # Compute left partial sum
    # saved_tensor[k+1] = x.T * G1 * ... * Gk
    # saved_tensor[k+1].shape: num_batch * (i1*...*ik-1) * ik * (jk+1*...*jd) * rk
    saved_tensors = []
    current_i_dim = 1
    saved_tensors.append(
        input_tensor.reshape(num_batch, current_i_dim, 1, -1, tensor.rank[0]))

    for k in range(order):
        current_tensor = saved_tensors[k]
        saved_tensors.append(
            torch.einsum(
                'aibcd,dbef->aiecf',
                current_tensor.reshape(num_batch, current_i_dim,
                                       tensorized_shape_x[k], -1,
                                       tensor.rank[k]), tensor.factors[k]))
        current_i_dim *= tensorized_shape_y[k]

    # Forward Pass
    # y[i1,...,id] = sum_j1_..._jd G1[i1,j1] * G2[i2,j2] * ... * Gd[id,jd] * x[j1,...,jd]
    output = saved_tensors[order].reshape(num_batch, -1)
    return output, saved_tensors


def ttm_times_matrix_bwd(tensor, dy, saved_tensors):
    """
    This function takes low-rank tensor, the gradient with respect to the output,
    and the tensors you saved, then computes the gradients with respect to the factors (grads) and the input tensor (dx)

    Parameters
    ----------
    tensor : BlockTT
        TTM tensorized weight matrix
    dy : 2D Tensor
        gradient dL/dy
    saved_tensors
        tensorized input matrix

    Returns
    -------
    grads
        gradients with respect to the factors
        equivalent to grads = [x.grad for x in tensor.factors]
    dx
        gradients with respect to the input tensor
        equivalent to input_mat.grad.detach()
    """

    grads = [
        torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for x in tensor.factors
    ]
    dx = torch.zeros(saved_tensors[0].shape,
                     dtype=saved_tensors[0].dtype,
                     device=saved_tensors[0].device)

    # Prepare tensor shape
    input_tensor = saved_tensors[0]
    num_batch = input_tensor.shape[0]
    tensorized_shape_x, tensorized_shape_y = tensor.tensorized_shape
    order = len(tensor.factors)

    j_right = 1
    i_left = prod(tensorized_shape_y)

    for k in range(order - 1, -1, -1):
        j_right *= tensorized_shape_x[k]
        i_left //= tensorized_shape_y[k]

        left = saved_tensors[k]
        cur_j_right = j_right
        cur_i_right = 1

        for cur_k in range(order - 1, k, -1):
            cur_j = tensorized_shape_x[cur_k]
            cur_i = tensorized_shape_y[cur_k]
            cur_j_right //= cur_j
            left = torch.einsum(
                'abicdeh,fdgh->abgicef',
                left.reshape(num_batch, i_left, cur_i_right, cur_j_right,
                             cur_j, tensor.rank[k], tensor.rank[cur_k + 1]),
                tensor.factors[cur_k])
            cur_i_right *= cur_i

        # Contract with dy
        grads[k] = torch.einsum(
            generate_contraction_string(k, order),
            left.reshape(
                (num_batch, ) + tensorized_shape_y[:k] +
                tensorized_shape_y[k + 1:] + (
                    cur_j_right,
                    tensor.rank[k],
                    tensor.rank[k + 1],
                )
            ),  # Shape: num_batch * i1 * ... * ik-1 * ik+1 * ... id * jk * rk * rk+1
            dy.reshape((num_batch, ) + tensorized_shape_y))

    # Compute dx
    saved_dx_tensors = []
    i_right = prod(tensorized_shape_y)
    j_left = 1
    saved_dx_tensors.append(dy.reshape((num_batch, ) + tensorized_shape_y))

    for k in range(order):
        i_right //= tensorized_shape_y[k]
        saved_dx_tensors.append(
            torch.einsum(
                'xcijr,rdce->xijde',
                saved_dx_tensors[k].reshape(num_batch, tensorized_shape_y[k],
                                            i_right, j_left, tensor.rank[k]),
                tensor.factors[k]).reshape(num_batch, i_right, -1,
                                           tensor.rank[k + 1]))
        j_left *= tensorized_shape_x[k]
    dx = saved_dx_tensors[-1].squeeze().permute(1, 0)

    return grads, dx


def generate_contraction_string(k, order):
    left = k
    right = order - k - 1
    CHAR = 'abcdefghijklmnopqrstuvwxyz'
    string = 'X' + CHAR[:left] + CHAR[
        left + 1:right + left +
        1] + 'JRS,X' + CHAR[:order] + '->RJ' + CHAR[k] + 'S'
    return string


# Elementwise fwd
def ttm_times_matrix_fwd_loop(tensor, matrix):
    """
    This function takes the input tensor "tensor", the input matrix "matrix"
    and returns tensor times matrix as well as any extra tensors you decide to save 
    for the backward pass

    Parameters
    ----------
    tensor : BlockTT
        TTM tensorized weight matrix
    matrix : Parameter object
        input matrix x

    Returns
    -------
    output
        tensor times matrix
        equivalent to matrix.T@tensor.to_matrix()
    saved_tensors
        tensorized input matrix
    """

    # Prepare tensor shape
    shape_x = matrix.shape
    shape_y = tuple([tensor.shape[1], matrix.shape[1]])
    tensorized_shape_x, tensorized_shape_y = tensor.tensorized_shape
    tensorized_shape_x += (shape_x[1], )
    tensorized_shape_y += (shape_y[1], )

    # Reshape input matrix to input tensor
    input_tensor = torch.reshape(matrix, tensorized_shape_x)

    # Forward Pass
    # y[i1,...,id] = sum_j1_j2_..._jd G_1[i1,j1] * G_2[i2,j2] * ... * G_d[id,jd] * x[j1,...,jd]

    output = torch.zeros(tensorized_shape_y, dtype=torch.float64)

    for i in itertools.product(*[range(y) for y in tensorized_shape_y[:-1]]):
        sum = 0
        for j in itertools.product(
                *[range(x) for x in tensorized_shape_x[:-1]]):
            product = torch.tensor([1], dtype=torch.float64)
            for index, j_index in enumerate(j):
                product = torch.matmul(
                    product, tensor.factors[index][:, j_index, i[index], :])
            sum += product * input_tensor[j]
        output[i] = sum

    # Reshape output tensor to output matrix
    output = torch.reshape(output, shape_y)
    output = torch.transpose(output, 0, 1)

    saved_tensors = input_tensor
    return output, saved_tensors


# Elementwise bwd
def ttm_times_matrix_bwd_loop(tensor, dy, saved_tensors):
    """
    This function takes low-rank tensor, the gradient with respect to the output, 
    and the tensors you saved, then computes the gradients with respect to the factors (grads) and the input tensor (dx)

    Parameters
    ----------
    tensor : BlockTT
        TTM tensorized weight matrix
    dy : 2D Tensor
        gradient dL/dy
    saved_tensors
        tensorized input matrix

    Returns
    -------
    grads
        gradients with respect to the factors
        equivalent to grads = [x.grad for x in tensor.factors]
    dx
        gradients with respect to the input tensor
        equivalent to input_mat.grad.detach()
    """

    # Prepare tensor shape
    input_tensor = saved_tensors
    tensorized_shape_x, tensorized_shape_y = tensor.tensorized_shape

    # Reshape dy matrix to tensor
    dy = dy.reshape((dy.shape[0], ) + tensorized_shape_y)

    order = len(tensorized_shape_x)
    tensorized_shape_x += (dy.shape[0], )
    tensorized_shape_y += (dy.shape[0], )

    # Compute dG
    grads = [torch.zeros(x.shape, dtype=torch.float64) for x in tensor.factors]

    # Permute tensor y
    for i in itertools.product(*[range(y) for y in tensorized_shape_y[:-1]]):

        # Compute Partial Sums
        # R_k[j1,...jk,ik+1,...id] = sum_jk+1_..._jd Gk+1[ik+1,jk+1] * ... * Gd[id,jd] x[j1,...,jd]

        R = [[] for idx in range(order)]

        for k in range(order - 1, -1, -1):

            R_k = torch.zeros(tensorized_shape_x[:k - order] + (
                1,
                tensor.rank[k + 1],
                dy.shape[0],
            ),
                              dtype=torch.float64)

            # Permute tensor x
            for j in itertools.product(
                    *[range(x) for x in tensorized_shape_x[:k - order]]):

                # Compute Corresponding Rk
                if k == order - 1:
                    R_k[j] = input_tensor[j]
                else:
                    sum = 0
                    for j_k in range(tensorized_shape_x[k + 1]):
                        sum += torch.tensordot(
                            tensor.factors[k + 1][:, j_k, i[k + 1], :],
                            torch.squeeze(R[k + 1][j + (j_k, )], 0),
                            dims=1)
                    R_k[j] = sum.reshape(
                        tuple([1, tensor.rank[k + 1], dy.shape[0]]))

            R[k] = R_k

        # Compute dy/dG
        # dy/dG[ik,jk] = sum_j1_..._jk-1 Gk-1[ik-1,jk-1].T * ... * G1[i1,j1].T * R[j1,j2,...,jk]

        for k in range(order):

            if k == 0:
                partial_grads = torch.tensordot(R[k],
                                                dy[(slice(None), ) + i],
                                                dims=1)

                for j_k in range(tensorized_shape_x[k]):
                    grads[k][:, j_k, i[k], :] += partial_grads[j_k, :, :]

            else:
                for j_k in range(tensorized_shape_x[k]):

                    for j in itertools.product(
                            *[range(x) for x in tensorized_shape_x[:k]]):
                        core_product = R[k][j][j_k]

                        cur_k = 0
                        while cur_k < k:
                            core_product = torch.tensordot(torch.transpose(
                                tensor.factors[cur_k][:, j[cur_k],
                                                      i[cur_k], :], 0, 1),
                                                           core_product,
                                                           dims=1)
                            cur_k += 1

                        # grads[k][:,j_k,i[k],:] += torch.tensordot(core_product,dy[(slice(None),)+i],dims=1)
                        tmp_grads = grads[k][:, j_k, i[k], :]
                        tmp_result = torch.tensordot(core_product,
                                                     dy[(slice(None), ) + i],
                                                     dims=1)
                        grads[k][:, j_k, i[k], :] = tmp_grads + tmp_result

    # Compute dx
    dx = torch.zeros(tensorized_shape_x, dtype=torch.float64)

    for j in itertools.product(*[range(x) for x in tensorized_shape_x[:-1]]):
        sum = 0
        for i in itertools.product(
                *[range(y) for y in tensorized_shape_y[:-1]]):
            product = torch.tensor([1], dtype=torch.float64)
            for index, i_index in enumerate(i):
                product = torch.matmul(
                    product, tensor.factors[index][:, j[index], i_index, :])
            sum += product * dy[(slice(None), ) + i]
        dx[j] = sum

    dx = dx.view(-1, dy.shape[0])
    return grads, dx
