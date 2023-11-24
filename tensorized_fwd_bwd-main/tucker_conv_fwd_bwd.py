"""
Tucker forward and backward pass convolution
"""
import tltorch
import torch
import torch.nn.functional as F
import argparse
import itertools
import warnings
import numpy as np

#Author Zi Yang

def tucker_conv_fwd(tensor, input, order):
    """
    Tucker conv forward pass
    """
    #you must replace the internals
    with torch.no_grad():


        size = list(tensor.shape)
        N = len(size)

        rank = [x.shape[1] for x in tensor.factors]

        out_channel = int(np.prod(size[0:order]))
        in_channel = input.shape[1]
        mat_size = list(input.shape[2:])
        minibatch = input.shape[0]

        kernel_size = [size[-2],size[-1]]

        out_channel_shape = size[0:order]
        in_channel_shape = size[order:2*order]

        output_size = [minibatch, out_channel] + mat_size



        output = input.reshape([minibatch] + in_channel_shape + mat_size)

        for A in tensor.factors[order:-2]:
            output = torch.tensordot(output, A, [[1],[0]])
        output = output.reshape([minibatch]  + mat_size + [np.prod(rank[order:-2])])
        output = torch.movedim(output, [1,2],[-2,-1])

        

        tmp = tensor.core

        for A in tensor.factors[:order]:
            tmp = torch.tensordot(tmp, A, [[0],[1]])
        tmp = torch.tensordot(tmp, tensor.factors[-2], [[order],[1]])
        tmp = torch.tensordot(tmp, tensor.factors[-1], [[order],[1]])




        tmp = tmp.reshape([np.prod(rank[order:-2]), out_channel] + kernel_size)
        tmp = torch.movedim(tmp,1,0)

        output = torch.nn.functional.conv2d(output,tmp, bias=None, stride=1, padding='same', dilation=1, groups=1)            

    saved_tensors = [input]
    return output, saved_tensors


def tucker_conv_bwd(tensor, dy, saved_tensors):
    """
    This function takes low-rank tensor, the gradient with respect to the output, 
    and the tensors you saved, then computes the gradients with respect to the factors (grads) and the input tensor (dx)
    """
    #the example below is just a dummy, you have to compute your own gradients
    with torch.no_grad():

        input = saved_tensors[0]

        size = list(tensor.shape)
        N = len(size)
        rank = [x.shape[1] for x in tensor.factors]

        order = int(N/2-1)

        out_channel = int(np.prod(size[0:order]))
        in_channel = input.shape[1]
        mat_size = list(input.shape[2:])
        minibatch = input.shape[0]

        out_channel_shape = size[0:order]
        in_channel_shape = size[order:2*order]

        padding_left = int(np.floor((size[-1] - 1.0)/2))
        padding_right = int(np.floor((size[-1] - 1.0)/2))
        padding_top = int(np.floor((size[-2] - 1.0)/2))
        padding_bottom = int(np.floor((size[-2] - 1.0)/2))

        padding_vec = (padding_left,padding_right,padding_top,padding_bottom)

        kernel_size = [size[-2],size[-1]]




        input_unf = torch.nn.functional.conv2d(torch.movedim(torch.nn.functional.pad(input,padding_vec), 1, 0), torch.movedim(dy,1,0), bias=None, stride=1, padding=0, dilation=1, groups=1)
        input_unf = torch.movedim(input_unf,1,0).reshape(out_channel_shape + in_channel_shape + kernel_size)


        grads = []
        
        for j in range(N):
            grad = input_unf
            for A in tensor.factors[0:j]:
                grad = torch.tensordot(grad,A,[[0],[0]])
            for A in tensor.factors[j+1:]:
                grad = torch.tensordot(grad,A,[[1],[0]])
            grad = torch.tensordot(grad, tensor.core, [list(range(1,N)), list(range(0,j)) + list(range(j+1,N))])
            grads.append(grad)

        
        grad_core = input_unf
        for A in tensor.factors:
            grad_core = torch.tensordot(grad_core, A, [[0],[0]])

        dx = dy.reshape([minibatch] + out_channel_shape + mat_size)

        for A in tensor.factors[:order]:
            dx = torch.tensordot(dx, A, [[1],[0]])

        dx = dx.reshape([minibatch]  + mat_size + [np.prod(rank[:order])] )
        dx = torch.movedim(dx, [1,2],[-2,-1])
        # dx = unfold(torch.nn.functional.pad(dx,padding_vec)).reshape([minibatch] + rank[:order] + kernel_size + mat_size)

        tmp = tensor.core

        for A in tensor.factors[order:-2]:
            tmp = torch.tensordot(tmp, A, [[order],[1]])
        tmp = torch.tensordot(tmp, torch.flip(tensor.factors[-2],[0]), [[order],[1]])
        tmp = torch.tensordot(tmp, torch.flip(tensor.factors[-1],[0]), [[order],[1]])
        

        tmp = tmp.reshape([np.prod(rank[:order]), in_channel] + kernel_size)
        tmp = torch.movedim(tmp,1,0)

        dx = torch.nn.functional.conv2d(dx,tmp, bias=None, stride=1, padding='same', dilation=1, groups=1)

       


    return grads + [grad_core], dx


###########################################
# Only modify this block
# You replace these functions with your own
# This is just a dummy
###########################################
