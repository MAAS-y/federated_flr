import torch
import numpy as np
import torch.nn.functional as F

# Author: Alvin Liu


def tt_times_matrix_fwd(tensor, matrix):
    ndims = tensor.order
    d = int(ndims / 2)
    ranks = tensor.rank
    tt_shape = tensor.shape
    tt_shape_row = list(tt_shape[:d])
    tt_shape_col = list(tt_shape[d:])
    matrix_rows = matrix.shape[0]
    matrix_cols = matrix.shape[1]

    with torch.no_grad():
        saved_tensors = [matrix]
        left = []
        right = []

        output = tensor.factors[0].reshape(-1, ranks[1])
        left.append(output)
        for core in tensor.factors[1:d]:
            output = torch.tensordot(output, core, dims=([-1], [0]))
            left.append(output)

        output = F.linear(matrix.T, torch.movedim(output.reshape(np.prod(tt_shape_row), -1), -1, 0))
        saved_tensors.append(left)

        temp = tensor.factors[d]
        right.append(temp)
        for core in tensor.factors[d + 1:]:
            temp = torch.tensordot(temp, core, dims=([-1], [0]))
            right.append(temp)

        output = F.linear(output, torch.movedim(temp.reshape(ranks[d], np.prod(tt_shape_col)),
                                                0, -1)).reshape(matrix_cols, np.prod(tt_shape_col))
        saved_tensors.append(right)

    return output, saved_tensors


def tt_times_matrix_bwd(tensor, dy, saved_tensors):
    ndims = tensor.order
    d = int(ndims / 2)
    ranks = tensor.rank
    tt_shape = tensor.shape
    tt_shape_row = list(tt_shape[:d])
    tt_shape_col = list(tt_shape[d:])
    matrix = saved_tensors[0]
    left = saved_tensors[1]
    right = saved_tensors[2]
    left_grads = []
    right_grads = []

    with torch.no_grad():
        dy_core_prod = right[-1]
        dy_core_prod = torch.tensordot(dy, dy_core_prod.reshape(dy_core_prod.shape[0], -1), dims=([1], [1]))
        matrix_dy_core_prod = torch.tensordot(matrix, dy_core_prod, dims=([1], [0]))

        for i in reversed(range(1, d)):
            grad = torch.tensordot(left[i-1].reshape(-1, ranks[i]),
                                   matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1,
                                                               ranks[d]),
                                   dims=([0], [0]))
            if i == d - 1:
                right_core = tensor.factors[i]
            else:
                grad = torch.tensordot(grad, right_core, dims=([2, 3], [1, 2]))
                right_core = torch.tensordot(tensor.factors[i], right_core,
                                             dims=([-1], [0])).reshape(ranks[i], -1, ranks[d])
            if grad.shape != tensor.factors[i].shape:
                grad = grad.reshape(list(tensor.factors[i].shape))

            left_grads.append(grad)

        left_grads.append(torch.tensordot(matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]),
                                          right_core, dims=([1, 2], [1, 2])).reshape(1, tt_shape_row[0], -1))

        left_grads = left_grads[::-1]

        matrix_core_prod = left[-1]
        matrix_core_prod = torch.tensordot(matrix_core_prod.reshape(-1, matrix_core_prod.shape[-1]),
                                           matrix, dims=([0], [0]))
        matrix_dy_core_prod = torch.tensordot(matrix_core_prod, dy, dims=([1], [0]))

        for i in reversed(range(1, d)):
            grad = torch.tensordot(right[i-1].reshape(-1, ranks[d+i]),
                                   matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i+1:]))),
                                   dims=([0], [0]))

            if i == d-1:
                right_core = tensor.factors[d+i].reshape(-1, tt_shape_col[i])
            else:
                grad = torch.tensordot(grad, right_core, dims=([-1], [1]))
                right_core = torch.tensordot(tensor.factors[d+i], right_core, dims=([-1], [0])).reshape(ranks[d+i], -1)
            if grad.shape != tensor.factors[d+i].shape:
                grad = grad.reshape(list(tensor.factors[i].shape))

            right_grads.append(grad)

        right_grads.append(torch.tensordot(matrix_dy_core_prod.reshape(ranks[d], tt_shape_col[0], -1),
                                           right_core, dims=([-1], [1])))

        right_grads = right_grads[::-1]

        dx = tensor.factors[-1].reshape(ranks[-2], -1)
        for core in reversed(tensor.factors[d:-1]):
            dx = torch.tensordot(core, dx, dims=([-1], [0]))

        dx = torch.tensordot(dy, dx.reshape(-1, np.prod(tt_shape_col)), dims=([-1], [-1]))

        temp = tensor.factors[0].reshape(-1, ranks[1])
        for core in tensor.factors[1:d]:
            temp = torch.tensordot(temp, core, dims=([-1], [0]))

        dx = torch.tensordot(temp.reshape(np.prod(tt_shape_row), -1), dx, dims=([-1], [-1]))

    return left_grads+right_grads, dx
