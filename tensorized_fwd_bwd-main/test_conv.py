import tltorch
import torch
import torch.nn.functional as F
import argparse
import itertools
import warnings
import numpy as np
from tucker_conv_fwd_bwd import tucker_conv_fwd, tucker_conv_bwd

warnings.filterwarnings("ignore")
TEST_DEVICES = ['cuda', 'cpu']
DTYPE = torch.float64
RTOL = 1e-7
BATCH_SIZE = 16
TEST_WEIGHT_SIZES = [3, 7]
TEST_RANKS = [1, 5]
TEST_ORDERS = [1, 2]
TEST_INPUT_CHANNELS = [32, 64]
TEST_OUTPUT_CHANNELS = [64, 128]
TEST_INPUT_SIZES = [32, 128]

parser = argparse.ArgumentParser()
parser.add_argument('--tensor-type',
                    type=str,
                    choices=['CP', 'TT', 'Tucker', 'TTM'],
                    default='CP')
parser.add_argument('--test-grads', action='store_true', default=False)
args = parser.parse_args()


def test_tensor_conv(tensor, input, order):
    """
    This mirrors torch.nn.functional.conv2d
    https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    """
    full_conv_kernel = tensor.to_tensor()

    weight_shape = list(tensor.shape)

    #if factorizing along input/output channels we need to reshape
    if tensor.name == 'BlockTT':  #TTM case since actual TT is just TT, not BlockTT
        full_conv_kernel = full_conv_kernel.reshape(weight_shape[0], -1,
                                                    *tensor.kernel_size)
    else:
        if order != 1:
            full_conv_kernel = full_conv_kernel.reshape(
                [np.prod(weight_shape[:order])] + weight_shape[order:])
            new_weight_shape = list(full_conv_kernel.shape)
            full_conv_kernel = full_conv_kernel.reshape(
                new_weight_shape[0], np.prod(new_weight_shape[1:order + 1]),
                *new_weight_shape[order + 1:])
    return F.conv2d(input,
                    full_conv_kernel,
                    bias=None,
                    stride=1,
                    padding='same',
                    dilation=1,
                    groups=1)



def get_tensor(size, order, rank, tensor_type):
    """
    This function returns a tensor of the specified type and rank
    """
    if tensor_type == 'TTM':
        print(size)
        #match Garipov et al. 2016 with "dummy" G_0 core
        #tmp = [size[0], size[1] * size[2] * size[3]]
        reshaped_dims = tltorch.utils.get_tensorized_shape(
            in_features=size[0],
            out_features=size[1],
            order=order + 1,  #avoid basic SVD
            verbose=False,
        )
        tensorized_dims = [[1,*reshaped_dims[0]],[size[2]*size[3],*reshaped_dims[1]]]  #[*reshaped_dims[0],*reshaped_dims[1]]
        tensor_type = 'blocktt'

        tensor = tltorch.TensorizedTensor.new(tensorized_shape=tensorized_dims,
                                              rank=rank,
                                              factorization=tensor_type,
                                              DTYPE=DTYPE)
        setattr(tensor, 'kernel_size', (size[2], size[3]))
    else:
        if order == 1:
            tensorized_dims = size
        else:
            reshaped_dims = tltorch.utils.get_tensorized_shape(
                in_features=size[0],
                out_features=size[1],
                order=order,
                verbose=False)
            tensorized_dims = [*reshaped_dims[0], *reshaped_dims[1], *size[2:]]

        tensor = tltorch.FactorizedTensor.new(shape=tensorized_dims,
                                              rank=rank,
                                              factorization=tensor_type,
                                              dtype=DTYPE)

    tensor = tensor.to(DTYPE)
    #init necessary, otherwise nan entries
    tltorch.tensor_init(tensor)

    return tensor


###########################################
# Only modify this block
# You replace these functions with your own
# This is just a dummy
###########################################
def my_tensor_conv(tensor, input, order):
    """
    This function takes the input weight tensor "tensor", the input tensor "input"
    and returns tensor convolution as well as any extra tensors you decide to save 
    for the backward pass
    """
    if tensor.name == 'Tucker':
        return tucker_conv_fwd(tensor, input, order)
    """
    else:
        raise ValueError('Tensor type {} not supported'.format(tensor.name))
    """
    output = test_tensor_conv(tensor, input, order)
    saved_tensors = []
    return output, saved_tensors


def my_get_grads(tensor, dy, saved_tensors):
    """
    This function takes low-rank tensor, the gradient with respect to the output, 
    and the tensors you saved, then computes the gradients with respect to the factors (grads) and the input tensor (dx)
    """
    if tensor.name == 'Tucker':
        return tucker_conv_bwd(tensor, dy, saved_tensors)
    """
    else:
        raise ValueError('Tensor type {} not supported'.format(tensor.name))
    """
    #the example below is just a dummy, you have to compute your own gradients
    grads = [x.grad for x in tensor.factors]
    dx = torch.zeros([10, 10]).to(DTYPE).to(grads[0].device)
    return grads, dx


###########################################
# Only modify this block
# You replace these functions with your own
# This is just a dummy
###########################################

weight_sizes = list(
    itertools.product(TEST_OUTPUT_CHANNELS, TEST_INPUT_CHANNELS,
                      TEST_WEIGHT_SIZES))
input_sizes = list(itertools.product([BATCH_SIZE], TEST_INPUT_SIZES))
tensor_info = list(itertools.product(TEST_RANKS, TEST_ORDERS))

for weight_size, input_size, tensor_info, test_device in itertools.product(
        weight_sizes, input_sizes, tensor_info, TEST_DEVICES):
    rank, order = tensor_info
    torch.manual_seed(0)
    weight_dims = [*weight_size] + [weight_size[-1]]
    input_dims = [input_size[0], weight_dims[1], input_size[1], input_size[1]]
    print(
        "Testing tensorized conv of size {} order {}, rank {}, and device {}".
        format(weight_dims, order, rank, test_device))
    print("\t Input size {}".format(input_dims))

    #construct low rank tensor and factors
    low_rank_tensor = get_tensor(weight_dims, order, rank,
                                 args.tensor_type).to(test_device)

    input_tensor = torch.nn.Parameter(torch.randn(input_dims,
                                                  dtype=DTYPE,
                                                  device=test_device),
                                      requires_grad=True)

    #compute test output after full tensor reconstruction
    test_output = test_tensor_conv(low_rank_tensor, input_tensor, order)

    #computes your output without tensor reconstruction
    output, saved_tensors = my_tensor_conv(low_rank_tensor, input_tensor,
                                           order)

    if torch.allclose(test_output, output, atol=0, rtol=RTOL):
        print("\tPassed forward test")
        pass
    else:
        max_relative_error = ((test_output - output).abs() /
                              test_output.abs()).max()
        raise ValueError("Failed forward test, relative error {}".format(
            max_relative_error))

    #only test grads if input flag true
    if args.test_grads:
        #use the squared norm to test backprop
        squared_norm = test_output.norm()**2
        squared_norm.backward()

        #gradient with respect to the output
        dy = 2.0 * test_output.detach().clone()
        test_dx = input_tensor.grad.detach()
        test_grads = [x.grad for x in low_rank_tensor.factors]
        if low_rank_tensor.name == 'Tucker':
            core_grad = low_rank_tensor.core.grad.detach()
            test_grads.append(core_grad)

        #you provide the gradients with respect to the factors and input
        grads, dx = my_get_grads(low_rank_tensor, dy, saved_tensors)

        for grad, test_grad in zip(grads + [dx], test_grads + [test_dx]):
            if not torch.allclose(grad, test_grad, atol=0, rtol=RTOL):
                max_relative_error = ((test_grad - grad).abs() /
                                      test_grad.abs()).max()
                raise ValueError(
                    "Failed backward test, relative error {}".format(
                        max_relative_error))
        print("\tPassed backward test")
    else:
        print("\tNot testing grads")

##########################
# Sample parameter access
##########################
#all factors
print("Factor shapes", [x.shape for x in low_rank_tensor.factors])
if args.tensor_type == 'Tucker':  #tucker core
    print("Tucker core shape", [low_rank_tensor.core.shape])
