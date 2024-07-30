#!/bin/bash

mkdir -p log_back/
mv log/* log_back/

echo "test1 start"
#/root/miniconda3/bin/python 1_column_slice_DCGG_GCN.py
#/root/miniconda3/bin/python 1_column_slice_DCGG_GIN.py
echo "test1 end"

echo "test2 start"
#/root/miniconda3/bin/python 2_dynamic_warp_DCGG_GCN.py
#/root/miniconda3/bin/python 2_dynamic_warp_DCGG_GIN.py
echo "test2 end"

echo "test3 start"
#/root/miniconda3/bin/python 3_dim_slice_DCGG_GCN.py
/root/miniconda3/bin/python 3_dim_slice_DCGG_GIN.py
echo "test3 end"


echo "test4 start"
/root/miniconda3/bin/python 4_hidden_acc_DCGG_GCN.py
/root/miniconda3/bin/python 4_hidden_acc_DCGG_GIN.py
echo "test4 end"