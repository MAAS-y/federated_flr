"""
Run CP format tensor times matrix forward
"""

import torch

# Author: Christian Lee

def cp_times_matrix_fwd(tensor, matrix):
    """
    Multiplies a tensorly CP tensorized matrix and an input matrix
    
    X^T @ W
    """
    
    saved_tensors = []
    order = len(tensor.tensorized_shape[0])
    
    # tensorize the input
    output = matrix.reshape(tensor.tensorized_shape[0] + 
                            (matrix.shape[1],))
    saved_tensors.append(output)
    
    # forward propagate with input factors
    output = torch.einsum('a...n,ar->...nr', output, tensor.factors[0])
    saved_tensors.append(output)
    for factor in tensor.factors[1:order]:
        output = torch.einsum('a...nr,ar->...nr', output, factor)
        saved_tensors.append(output)
    
    # forward propagate with output factors
    for factor in tensor.factors[order:tensor.order-1]:
        output = torch.einsum('n...r,ar->n...ar', output, factor)
        saved_tensors.append(output)
    output = torch.einsum('n...r,ar->n...a', output, tensor.factors[-1])
    
    # vectorize the output
    output = output.reshape((matrix.shape[1], tensor.shape[1]))
    
    return output, saved_tensors

def cp_times_matrix_bwd(tensor, dy, saved_tensors):
    '''
    X^T @ W backprob
    '''

    grads = []

    dy = dy.reshape((dy.shape[0],) + tensor.tensorized_shape[1])

    grads.append(torch.einsum('...a,...r->ar', dy, saved_tensors[-1]))
    dy = torch.einsum('...a,ar->...r', dy, tensor.factors[-1])

    order = len(tensor.tensorized_shape[0])

    for (factor, saved_tensor) in zip(reversed(tensor.factors[order:tensor.order-1]), 
                                      reversed(saved_tensors[order:tensor.order-1])):  
        grads.append(torch.einsum('...ar,...r->ar', dy, saved_tensor))
        dy = torch.einsum('...ar,ar->...r', dy, factor)

    for (factor, saved_tensor) in zip(reversed(tensor.factors[1:order]), 
                                      reversed(saved_tensors[1:order])):
            grads.append(torch.einsum('...r,a...r->ar', dy, saved_tensor))
            dy = torch.einsum('...r,ar->a...r', dy, factor)

    grads.append(torch.einsum('...r,a...->ar', dy, saved_tensors[0]))
    dy = torch.einsum('...nr,ar->a...n', dy, tensor.factors[0])

    dy = dy.reshape((tensor.shape[0], saved_tensors[0].shape[-1]))

    grads = [x for x in reversed(grads)]
    
    return grads, dy

def cp_times_matrix_fwd_2(tensor, matrix):
    """
    Multiplies a tensorly CP tensorized matrix and an input matrix
    
    X @ W
    """
    
    order = len(tensor.tensorized_shape[0])
    saved_tensors = []

    # tensorize the input
    output = matrix.reshape((matrix.shape[0],) + tensor.tensorized_shape[0])
    saved_tensors.append(output)

    # forward propagate with input factors
    output = torch.einsum('na...,ar->n...r', output, tensor.factors[0])
    saved_tensors.append(output)
    for factor in tensor.factors[1:order]:
        output = torch.einsum('na...r,ar->n...r', output, factor)
        saved_tensors.append(output)

    # forward propagate with output factors
    for factor in tensor.factors[order:tensor.order-1]:
        output = torch.einsum('n...r,ar->n...ar', output, factor)
        saved_tensors.append(output)
    output = torch.einsum('n...r,ar->n...a', output, tensor.factors[-1])
    
    # vectorize the output
    output = output.reshape((matrix.shape[0], tensor.shape[1]))
    
    return output, saved_tensors

def cp_times_matrix_bwd_2(tensor, grad, saved_tensors):
    '''
    X @ W backprob
    '''
    
    order = len(tensor.tensorized_shape[0])
    factor_grads = []

    # derivative of reshape
    grad = grad.reshape((grad.shape[0],) + tensor.tensorized_shape[1])

    # derivatives for 'n...r,ar->n...a'
    factor_grads.append(torch.einsum('...a,...r->ar', grad, saved_tensors[-1]))
    grad = torch.einsum('...a,ar->...r', grad, tensor.factors[-1])

    for (factor, saved_tensor) in zip(reversed(tensor.factors[order:tensor.order-1]), 
           reversed(saved_tensors[order:tensor.order-1])):     
        # derivatives for 'n...r,ar->n...ar'
        factor_grads.append(torch.einsum('...ar,...r->ar', grad, saved_tensor))
        grad = torch.einsum('...ar,ar->...r', grad, factor)

    for (factor, saved_tensor) in zip(reversed(tensor.factors[1:order]), 
           reversed(saved_tensors[1:order])):
        # derivatives for 'na...r,ar->n...r'
        factor_grads.append(torch.einsum('n...r,na...r->ar', grad, saved_tensor))
        grad = torch.einsum('n...r,ar->na...r', grad, factor)

    # derivatives for 'na...,ar->n...r'
    factor_grads.append(torch.einsum('n...r,na...->ar', grad, saved_tensors[0]))
    grad = torch.einsum('n...r,ar->na...', grad, tensor.factors[0])

    # derivative for reshape
    grad = grad.reshape((tensor.shape[1], tensor.shape[0]))

    factor_grads = [x for x in reversed(factor_grads)]
    
    return factor_grads, grad