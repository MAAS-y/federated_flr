import tltorch
import torch
import argparse
import itertools
import warnings
import numpy as np
from fc_utils import my_get_grads#my_tensor_times_matrix, my_get_grads
from tensorized_fwd_bwd.tensorized_fwd_bwd.tensor_times_matrix import tensor_times_matrix_fwd


warnings.filterwarnings("ignore")

DTYPE = torch.float64
TEST_DEVICE = ['cpu', 'cuda']
RTOL = 1e-7
TEST_SIZES = [128, 256, 512]
TEST_RANKS = [1, 3, 5]
TEST_ORDERS = [2, 3, 4]

parser = argparse.ArgumentParser()
parser.add_argument('--tensor-type',
                    type=str,
                    choices=['CP', 'TT', 'TTM', 'Tucker'],
                    default='CP')
parser.add_argument('--test-grads', action='store_true', default=False)
args = parser.parse_args()


def test_tensor_times_matrix(tensor, matrix):
    """
    This mirrors torch.nn.functional.linear y = x^T W
    """
    if tensor.name == 'TT':
        return matrix.T @ tensor.to_tensor().reshape(matrix.shape[0], -1)
    else:
        return matrix.T @ tensor.to_matrix()


def get_tensor(size, order, rank, tensor_type):
    """
    This function returns a tensor of the specified type and rank
    """
    tensorized_dims = tltorch.utils.get_tensorized_shape(in_features=size[0],
                                                         out_features=size[1],
                                                         order=order,
                                                         verbose=False)

    if tensor_type == 'TT':
        tensor_type = 'tensor_train'
        tensorized_dims = [*tensorized_dims[0], *tensorized_dims[1]]
        tensor = tltorch.TTTensor.new(shape=tensorized_dims,
                                      rank=rank,
                                      dtype=DTYPE)

    else:
        if tensor_type == 'TTM':
            tensor_type = 'blocktt'

        tensor = tltorch.TensorizedTensor.new(tensorized_shape=tensorized_dims,
                                              rank=rank,
                                              factorization=tensor_type,
                                              DTYPE=DTYPE)
        tensor = tensor.to(dtype=DTYPE)

    #init necessary, otherwise nan entries
    tltorch.tensor_init(tensor)

    return tensor



test_size_pairs = list(itertools.product(TEST_SIZES, TEST_SIZES))
loop_product = itertools.product(test_size_pairs, TEST_RANKS, TEST_ORDERS,
                                 TEST_DEVICE)

for test_size, test_rank, test_order, test_device in loop_product:
    torch.manual_seed(0)
    print("Testing tensorized matrix of size {}, order {}, rank {}, device {}".
          format(test_size, test_order, test_rank, test_device))

    #construct low rank tensor and factors
    low_rank_tensor = get_tensor(test_size, test_order, test_rank,
                                 args.tensor_type).to(device=test_device)

    input_mat = torch.nn.Parameter(torch.randn(
        test_size, dtype=DTYPE).to(device=test_device),
                                   requires_grad=True)

    #compute test output after full tensor reconstruction
    test_output = test_tensor_times_matrix(low_rank_tensor, input_mat)

    #computes your output without tensor reconstruction
    if args.test_grads:
        output, saved_tensors = tensor_times_matrix_fwd(low_rank_tensor, input_mat, True)
    else:
        output = tensor_times_matrix_fwd(low_rank_tensor, input_mat, return_saved_tensors=args.test_grads)

    if torch.allclose(test_output, output, atol=0, rtol=RTOL):
        print("\tPassed forward test")
        pass
    else:
        max_relative_error = ((test_output - output).abs() /
                              test_output.abs()).max()
        raise ValueError("Failed forward test, max relative error {}".format(
            max_relative_error.item()))

    #only test grads if input flag true
    if args.test_grads:
        #use the squared norm to test backprop
        squared_norm = test_output.norm()**2
        squared_norm.backward()

        test_dx = input_mat.grad.detach()
        test_grads = [x.grad for x in low_rank_tensor.factors]
        #gradient with respect to the output
        dy = 2.0 * test_output.detach().clone()

        if low_rank_tensor.name == 'Tucker':
            core_grad = low_rank_tensor.core.grad.detach()
            test_grads.append(core_grad)

        #you provide the gradients with respect to the factors and input
        grads, dx = my_get_grads(low_rank_tensor, dy, saved_tensors)

        for grad, test_grad in zip(grads + [dx], test_grads + [test_dx]):
            if not torch.allclose(grad, test_grad, atol=0, rtol=RTOL):
                max_relative_error = ((grad - test_grad).abs() /
                                      test_grad.abs()).max()
                raise ValueError(
                    "Failed backward test, max relative error {}".format(
                        max_relative_error.item()))

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
