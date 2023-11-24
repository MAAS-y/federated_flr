from math import prod
import torch

# Author Xinling Yu

def ttm_conv_fwd(tensor, input, order):
    
    shape = input.shape
    batch = shape[0]
    H = shape[2]
    W = shape[3]
    
    
    # shape num_batch*l^2c*HW
    input_unf = torch.nn.functional.unfold(input, (tensor.kernel_size[0], tensor.kernel_size[1]), padding=int((tensor.kernel_size[0]-1)/2))
    
    # shape num_batch*HW*l^2c
    input_unf = input_unf.permute(0,2,1)

    shape_x = input_unf.shape
    tensorized_shape_y, tensorized_shape_x = tensor.tensorized_shape
    num_batch = shape_x[0]*shape_x[1]
    order = len(tensor.factors)
    

    input_tensor = torch.reshape(input_unf, [num_batch,] + tensorized_shape_x)
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
                                       tensor.rank[k]), tensor.factors[k].permute(0,2,1,3)))
        current_i_dim *= tensorized_shape_y[k]

   
    output = saved_tensors[order].reshape(num_batch, -1)
    output = output.reshape(batch, H, W, tensor.shape[0]).permute(0,3,1,2)
    return output, saved_tensors


def ttm_conv_bwd(tensor, dy, saved_tensors):
    
    
    batch = dy.shape[0]
    S = dy.shape[1]
    H = dy.shape[2]
    W = dy.shape[3]
    dy = dy.permute(0,2,3,1).reshape(-1,S)
    grads = [
        torch.zeros(x.permute(0,2,1,3).shape, dtype=x.dtype, device=x.device)
        for x in tensor.factors
    ]
    dx = torch.zeros(saved_tensors[0].shape,
                     dtype=saved_tensors[0].dtype,
                     device=saved_tensors[0].device)

    input_tensor = saved_tensors[0]
    num_batch = input_tensor.shape[0]
    tensorized_shape_y, tensorized_shape_x = tensor.tensorized_shape
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
                tensor.factors[cur_k].permute(0,2,1,3))
            cur_i_right *= cur_i

        # Contract with dy
        grads[k] = torch.einsum(
            generate_contraction_string(k, order),
            left.reshape(
                [num_batch, ] + tensorized_shape_y[:k] +
                tensorized_shape_y[k + 1:] + [
                    cur_j_right,
                    tensor.rank[k],
                    tensor.rank[k + 1],
                ]
            ),  
            dy.reshape([num_batch, ] + tensorized_shape_y))
        grads[k] = grads[k].permute(0,2,1,3)

    # Compute dx
    saved_dx_tensors = []
    i_right = prod(tensorized_shape_y)
    j_left = 1
    saved_dx_tensors.append(dy.reshape([num_batch, ] + tensorized_shape_y))

    for k in range(order):
        i_right //= tensorized_shape_y[k]
        saved_dx_tensors.append(
            torch.einsum(
                'xcijr,rdce->xijde',
                saved_dx_tensors[k].reshape(num_batch, tensorized_shape_y[k],
                                            i_right, j_left, tensor.rank[k]),
                tensor.factors[k].permute(0,2,1,3)).reshape(num_batch, i_right, -1,
                                           tensor.rank[k + 1]))
        j_left *= tensorized_shape_x[k]
    dx = saved_dx_tensors[-1].squeeze()
    dx = dx.reshape(batch, H*W, -1).permute(0,2,1)
    dx = torch.nn.functional.fold(dx, (H,W), (tensor.kernel_size[0], tensor.kernel_size[1]), padding=int((tensor.kernel_size[0]-1)/2))
    return grads, dx

def generate_contraction_string(k, order):
    left = k
    right = order - k - 1
    CHAR = 'abcdefghijklmnopqrstuvwxyz'
    string = 'X' + CHAR[:left] + CHAR[
        left + 1:right + left +
        1] + 'JRS,X' + CHAR[:order] + '->RJ' + CHAR[k] + 'S'
    return string