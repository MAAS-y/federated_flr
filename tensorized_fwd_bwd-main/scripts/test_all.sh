#!/bin/bash

mkdir logs

echo "Testing forward only"
for tensor_type in 'CP' 'TT' 'TTM' 'Tucker';
do python test_fc.py --tensor-type $tensor_type > logs/test_fc_fwd_$tensor_type.log;
done

echo "Testing forward and backward"
for tensor_type in 'CP' 'TT' 'TTM' 'Tucker';
do python test_fc.py --tensor-type $tensor_type > logs/test_fc_fwd_bwd_$tensor_type.log;
done
