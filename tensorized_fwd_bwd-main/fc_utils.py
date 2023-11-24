import tltorch
import torch
import argparse
import itertools
import warnings
import numpy as np
from tt_fc_fwd_bwd import tt_times_matrix_fwd, tt_times_matrix_bwd
from ttm_fc_fwd_bwd import ttm_times_matrix_fwd, ttm_times_matrix_bwd
from tucker_fc_fwd_bwd import tucker_times_matrix_fwd, tucker_times_matrix_bwd
from cp_fc_fwd_bwd import cp_times_matrix_fwd, cp_times_matrix_bwd


###########################################
# Only modify this block
# You replace these functions with your own
# This is just a dummy
###########################################
def my_tensor_times_matrix(tensor, matrix):
    """
    This function takes the input tensor "tensor", the input matrix "matrix"
    and returns tensor times matrix as well as any extra tensors you decide to save 
    for the backward pass
    """
    if tensor.name == 'TT':
        return tt_times_matrix_fwd(tensor, matrix)
    elif tensor.name == 'BlockTT':
        return ttm_times_matrix_fwd(tensor, matrix)
    elif tensor.name == 'Tucker':
        return tucker_times_matrix_fwd(tensor, matrix)
    elif tensor.name == 'CP':
        return cp_times_matrix_fwd(tensor, matrix)
    else:
        raise ValueError('Unknown tensor type')
        #you must replace the internals here
        output = test_tensor_times_matrix(tensor, matrix)
        saved_tensors = []
    return output, saved_tensors


def my_get_grads(tensor, dy, saved_tensors):
    """
    This function takes low-rank tensor, the gradient with respect to the output, 
    and the tensors you saved, then computes the gradients with respect to the factors (grads) and the input tensor (dx)
    """
    if tensor.name == 'TT':
        return tt_times_matrix_bwd(tensor, dy, saved_tensors)
    elif tensor.name == 'BlockTT':
        return ttm_times_matrix_bwd(tensor, dy, saved_tensors)
    elif tensor.name == 'Tucker':
        return tucker_times_matrix_bwd(tensor, dy, saved_tensors)
    elif tensor.name == 'CP':
        return cp_times_matrix_bwd(tensor, dy, saved_tensors)
    else:
        raise ValueError('Unknown tensor type')
        #the example below is just a dummy, you have to compute your own gradients
        grads = [x.grad for x in tensor.factors]
        dx = torch.zeros([10, 10])
        return grads, dx


###########################################
# Only modify this block
# You replace these functions with your own
# This is just a dummy
###########################################
