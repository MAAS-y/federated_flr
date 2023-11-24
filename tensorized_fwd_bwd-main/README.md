## Setup

Create a conda environment and install the necessary packages:
```bash
conda create -n $NAME python=3.9
pip install -r requirements.txt
```
Right now the convolutions as well as custom gradient operations for the matrix multiply backward pass and the convolution backward pass are still under development. To install the unreconstructed forward propagation package:
```
cd tensorized_fwd_bwd
pip install -e .
cd ..
```
Then you can perform the tensor-times-matrix operation X^TA where X is a matrix with shape [M,N] and A is a tensorized matrix with shape [M,K] by calling 
```
out = tensorized_fwd_bwd.tensor_times_matrix_fwd(tensor,matrix)
```
This package assumes that you use [tensorly-torch](http://tensorly.org/torch/dev/) to create the tensor A.

## Running Tests

Your forward pass (tensor times matrix) must pass the tests in `test_fc.py`. You must implement the `my_tensor_times_matrix` and `my_get_grads` functions. The current implementations are just dummies.

First, make sure that your forward pass (without reconstruction) is correct by running 
```
python test_fc.py --tensor-type $TENSOR_TYPE
```
If any tests fail an error will be raised. 

Once your forward pass completes all tests, test your backward pass by adding the `--test-grads` flag.
```
python test_fc.py --tensor-type $TENSOR_TYPE --test-grads
```

## Misc

To access low-rank tensor parameters (factors and/or core) see the end of `test_fc.py`.

Important: your function signature must be the same as mine. You should only change the code in the functions `my_tensor_times_matrix` and `my_get_grads`.

It is possible that the relative tolerance `1e-7` is too strict. If your tests fail at `1e-7` you can relax it to `1e-6`.
