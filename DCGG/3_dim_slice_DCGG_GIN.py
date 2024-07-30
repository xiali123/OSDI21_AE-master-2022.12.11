#!/usr/bin/env python3
import os
import time
import sys
sys.path.append('../')
import common_dataset
os.environ["PYTHONWARNINGS"] = "ignore"

dataset=common_dataset.dataset


islocal = common_dataset.islocal
mytime = str(time.time())
outfile = './log/3_dim_slice_GIN_' + mytime +'.log'

run_GCN = False              # whether to run GCN model. 
enable_rabbit = False        # whether to enable rabbit reordering in auto and manual mode.
manual_mode = False         # whether to use the manually configure the setting.
verbose_mode = False         # whether to printout more information such as the layerwise parameter.
loadFromTxt = False         # whether to load data from a plain txt file.

is_sort = True
is_dynamic_warp = True
is_data_stream = True
is_dim_slice = True

max_warp_size = 1024
dim_slice_size = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 80, 96, 128]

if run_GCN:
    model = 'gcn'
    warpPerBlock = 8        # only effective in manual model
    hidden = [16] 
else:
    model = 'gin'
    warpPerBlock = 2        # only effective in manual model 2 for citeseer 6 for remaining datasets
    hidden = [64]

partsize_li = [16]          # only effective in manual model
partsize = 16

python_cmd = common_dataset.python_cmd
for cur_dim_slice in dim_slice_size:
    for hid in hidden:
        for data, d, c, load, warp_size_k in dataset:
            loadFromTxt = load
            outfile_new = outfile+"-"+data+".txt"
            command = "{} GNNA_main.py --dataset {} --dim {} --hidden {} \
                        --classes {} --partSize {} --model {} --warpPerBlock {}\
                        --manual_mode {} --verbose_mode {} --enable_rabbit {} --loadFromTxt {}  --warp_size_k {} --cur_dim_slice {}  --max_warp_size {} --is_sort {} --is_dynamic_warp {} --is_data_stream {} --is_dim_slice {} | tee -a {}"
            command = command.format(python_cmd, data, d, hid, c, partsize, model, warpPerBlock, \
                                     manual_mode, verbose_mode, enable_rabbit, loadFromTxt, warp_size_k, cur_dim_slice, max_warp_size, is_sort, is_dynamic_warp, is_data_stream, is_dim_slice, outfile)
            # command = "python GNNA_main.py -loadFromTxt --dataset {} --partSize {} --dataDir {}".format(data, partsize, '/home/yuke/.graphs/orig')
            os.system(command)

outfile = './log/3_dim_no_slice_GIN_' +mytime  +'.log'
is_dim_slice = False
for hid in hidden:
    for data, d, c, load, warp_size_k in dataset:
        loadFromTxt = load
        outfile_new = outfile+"-"+data+".txt"
        command = "{} GNNA_main.py --dataset {} --dim {} --hidden {} \
                    --classes {} --partSize {} --model {} --warpPerBlock {}\
                    --manual_mode {} --verbose_mode {} --enable_rabbit {} --loadFromTxt {}  --warp_size_k {} --max_warp_size {} --is_sort {} --is_dynamic_warp {} --is_data_stream {} --is_dim_slice {} | tee -a {}"
        command = command.format(python_cmd, data, d, hid, c, partsize, model, warpPerBlock, \
                                 manual_mode, verbose_mode, enable_rabbit, loadFromTxt, warp_size_k, max_warp_size, is_sort, is_dynamic_warp, is_data_stream, is_dim_slice, outfile)
        # command = "python GNNA_main.py -loadFromTxt --dataset {} --partSize {} --dataDir {}".format(data, partsize, '/home/yuke/.graphs/orig')
        os.system(command)