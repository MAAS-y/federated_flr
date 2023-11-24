import tltorch
import torch
import torch.nn.functional as F
import numpy as np

# Author: Christian Lee

def cp_conv_fwd(tensor, input_tensor, order):
    
    saved_tensors = []
    
    # tensorize the input
    output = input_tensor.reshape((input_tensor.shape[0],) + tensor.shape[order:-2] + input_tensor.shape[-2:])
    saved_tensors.append(output)
    
    # forward propagate with input factors
    output = torch.einsum('na...xy,ar->nr...xy', output, tensor.factors[order])
    saved_tensors.append(output)
    for factor in tensor.factors[order+1:order*2]:
        output = torch.einsum('nra...xy,ar->nr...xy', output, factor)
        saved_tensors.append(output)
    
    # x and y convolutions
    pad = int(tensor.shape[-2]/2)
    output = torch.nn.functional.conv2d(output, 
                                        tensor.factors[order*2].T.reshape(tensor.rank, 1, tensor.shape[-2], 1), 
                                        padding=(pad, 0), 
                                        groups=tensor.rank)
    saved_tensors.append(output)
    
    pad = int(tensor.shape[-1]/2)
    output = torch.nn.functional.conv2d(output, 
                                        tensor.factors[order*2+1].T.reshape(tensor.rank, 1, 1, tensor.shape[-1]), 
                                        padding=(0, pad), 
                                        groups=tensor.rank)
    saved_tensors.append(output)
    
    # forward propagate with output factors
    for factor in tensor.factors[:order-1]:
        output = torch.einsum('nr...xy,ar->nr...axy', output, factor)
        saved_tensors.append(output)
    output = torch.einsum('nr...xy,ar->n...axy', output, tensor.factors[order-1])
    
    # reshape the output
    output = output.reshape((output.shape[0], np.prod(tensor.shape[:order]), output.shape[-2], output.shape[-1]))
    
    return output, saved_tensors

def cp_conv_bwd(tensor, dy, saved_tensors):
    
    order = int((len(tensor.shape) - 2)/2)
    
    out_factor_grads = []
    
    dy = dy.reshape((saved_tensors[0].shape[0],) + tensor.shape[:order] + saved_tensors[0].shape[-2:])

    out_factor_grads.append(torch.einsum('n...axy,nr...xy->ar', dy, saved_tensors[-1]))
    dy = torch.einsum('n...axy,ar->nr...xy', dy, tensor.factors[order-1])

    for factor, saved_tensor in zip(reversed(tensor.factors[:order-1]), 
                                    reversed(saved_tensors[-order:-1])):
        out_factor_grads.append(torch.einsum('nr...axy,nr...xy->ar', dy, saved_tensor))
        dy = torch.einsum('nr...axy,ar->nr...xy', dy, factor)

    
    factor_grads = []
    pad = int(tensor.shape[-1]/2)
    factor_grads.append(F.conv3d(torch.einsum('ncxy->cnxy', saved_tensors[-order-1]).unsqueeze(0), 
                                 torch.einsum('ncxy->cnxy', dy).unsqueeze(1), 
                                 padding=(0,0,pad), 
                                 groups=tensor.rank).squeeze(0).reshape(tensor.rank, tensor.shape[-1]).T)
    dy = torch.nn.functional.conv_transpose2d(dy, 
                                              tensor.factors[-1].T.reshape(tensor.rank, 1, 1, tensor.shape[-1]),
                                              padding=(0,pad), 
                                              groups=tensor.rank)
    

    pad = int(tensor.shape[-2]/2)
    factor_grads.append(F.conv3d(torch.einsum('ncxy->cnxy', saved_tensors[-order-2]).unsqueeze(0), 
                                 torch.einsum('ncxy->cnxy', dy).unsqueeze(1), 
                                 padding=(0,pad,0), 
                                 groups=tensor.rank).squeeze(0).reshape(tensor.rank, tensor.shape[-2]).T)
    dy = torch.nn.functional.conv_transpose2d(dy, 
                                              tensor.factors[-2].T.reshape(tensor.rank, 1, tensor.shape[-2], 1),
                                              padding=(pad,0), 
                                              groups=tensor.rank)

    for factor, saved_tensor in zip(reversed(tensor.factors[order+1:order*2]), 
                                    reversed(saved_tensors[1:-order-2])):
        factor_grads.append(torch.einsum('nr...xy,nra...xy->ar', dy, saved_tensor))
        dy = torch.einsum('nr...xy,ar->nra...xy', dy, factor)

    factor_grads.append(torch.einsum('nr...xy,na...xy->ar', dy, saved_tensors[0]))
    dy = torch.einsum('nr...xy,ar->na...xy', dy, tensor.factors[order])
    
    dy = dy.reshape((saved_tensors[0].shape[0], 
                     np.prod(saved_tensors[0].shape[1:order+1])) + saved_tensors[0].shape[-2:])
    
    factor_grads = [x for x in reversed(out_factor_grads)] + [x for x in reversed(factor_grads)]
    
    return factor_grads, dy