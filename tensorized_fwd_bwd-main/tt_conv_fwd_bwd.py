import torch
import torch.nn.functional as F
import numpy as np


def tt_conv_fwd(kernel_tensor, input_tensor, order):
    """
    This function takes the input weight tensor "tensor", the input tensor "input"
    and returns tensor convolution as well as any extra tensors you decide to save
    for the backward pass
    """

    tensor_shape = list(kernel_tensor.shape)
    core_len = len(tensor_shape)
    ranks = kernel_tensor.rank

    batch = input_tensor.shape[0]
    height = input_tensor.shape[-2]
    width = input_tensor.shape[-1]

    channel_out_shape = tensor_shape[:order]
    channel_in_shape = tensor_shape[order:2 * order]
    channel_out = np.prod(channel_out_shape)
    channel_in = np.prod(channel_in_shape)

    l_1 = tensor_shape[-2]
    l_2 = tensor_shape[-1]
    kernel_size = [l_1, l_2]

    pad_l = int(np.floor((l_2 - 1) / 2))
    pad_r = int(np.floor((l_2 - 1) / 2))
    pad_t = int(np.floor((l_1 - 1) / 2))
    pad_b = int(np.floor((l_1 - 1) / 2))
    pad_size = (pad_l, pad_r, pad_t, pad_b)

    with torch.no_grad():
        output = input_tensor.reshape([batch] + channel_in_shape + [height, width])

        output = torch.tensordot(output, kernel_tensor.factors[order], dims=([1], [1]))
        if order != 1:
            for core in kernel_tensor.factors[order + 1:-2]:
                output = torch.tensordot(output, core, dims=([1, -1], [1, 0]))

        output = torch.permute(output, (0, 3, 4, 1, 2)).reshape(-1, ranks[2 * order], height, width)
        kernel = torch.tensordot(kernel_tensor.factors[-2], kernel_tensor.factors[-1], dims=([2], [0]))
        kernel = torch.movedim(kernel, -1, 0)
        output = F.conv2d(F.pad(output, pad_size), kernel, bias=None, stride=1, padding=0, dilation=1, groups=1)
        output = torch.movedim(output.reshape(batch, -1, height, width), 1, 0)

        for core in reversed(kernel_tensor.factors[:order]):
            output = torch.tensordot(core, output, dims=([2], [0]))

        output = torch.movedim(output.reshape(channel_out, batch, height, width), 1, 0)

        saved_tensors = [input_tensor]

    return output, saved_tensors


def tt_conv_bwd(kernel_tensor, dy, saved_tensors):
    """
    This function takes low-rank tensor, the gradient with respect to the output,
    and the tensors you saved, then computes the gradients with respect to the factors (grads) and the input tensor (dx)
    """
    # Get all the shapes
    tensor_shape = list(kernel_tensor.shape)
    core_len = len(tensor_shape)
    order = int((core_len - 2) / 2)
    ranks = kernel_tensor.rank
    batch = dy.shape[0]
    height = dy.shape[2]
    width = dy.shape[3]
    channel_out_shape = tensor_shape[:order]
    channel_in_shape = tensor_shape[order:2 * order]
    channel_out = np.prod(channel_out_shape)
    channel_in = np.prod(channel_in_shape)

    l_1 = tensor_shape[-2]
    l_2 = tensor_shape[-1]
    kernel_size = [l_1, l_2]

    pad_l = int(np.floor((l_2 - 1) / 2))
    pad_r = int(np.floor((l_2 - 1) / 2))
    pad_t = int(np.floor((l_1 - 1) / 2))
    pad_b = int(np.floor((l_1 - 1) / 2))
    pad_size = (pad_l, pad_r, pad_t, pad_b)

    input_tensor = saved_tensors[0]
    padded_input = torch.movedim(F.pad(input_tensor, pad_size), 1, 0)

    with torch.no_grad():
        grads = []

        input_dy_prod = F.conv2d(padded_input,
                                 torch.movedim(dy, 1, 0),
                                 bias=None, stride=1, padding=0, dilation=1, groups=1)
        input_dy_prod = torch.movedim(input_dy_prod, 1, 0).reshape(
            [1] + channel_out_shape + channel_in_shape + kernel_size + [1])

        for k in range(core_len):
            grad = input_dy_prod

            for core in kernel_tensor.factors[:k]:
                grad = torch.tensordot(core, grad, dims=([0, 1], [0, 1]))
            for core in reversed(kernel_tensor.factors[k + 1:]):
                grad = torch.tensordot(grad, core, dims=([-2, -1], [1, 2]))

            grads.append(grad)

        del input_dy_prod

        dx = dy.reshape([batch] + channel_out_shape + [height, width])
        dx = torch.tensordot(dx, kernel_tensor.factors[0], dims=([1], [1]))

        if order != 1:
            for core in kernel_tensor.factors[1:order]:
                dx = torch.tensordot(dx, core, dims=([1, -1], [1, 0]))

        dx = torch.permute(dx, (0, 3, 4, 1, 2)).reshape(batch, ranks[order], height, width)

        kernel = torch.tensordot(torch.flip(kernel_tensor.factors[-2], [1]),
                                 torch.flip(kernel_tensor.factors[-1], [1]),
                                 dims=([2], [0]))

        for core in reversed(kernel_tensor.factors[order:-2]):
            kernel = torch.tensordot(core, kernel, dims=([2], [0]))

        kernel = torch.movedim(kernel.reshape(ranks[order], channel_in, l_1, l_2), 1, 0)

        dx = F.conv2d(F.pad(dx, pad_size), kernel, bias=None, stride=1, padding=0, dilation=1, groups=1)

    return dx, grads
