#!/usr/bin/env python3
import os
import time
import sys
sys.path.append('../')
import common_dataset
os.environ["PYTHONWARNINGS"] = "ignore"

dataset=common_dataset.dataset

islocal = common_dataset.islocal
#islocal=True
if islocal:
    dataset = [
        ('ogbn-proteins'		, 100      , 47  , True, 15),
    ]


outfile = './log/0_bench_DCGG_GCN_' + str(time.time())+".log"

run_GCN = True              # whether to run GCN model. 
enable_rabbit = False        # whether to enable rabbit reordering in auto and manual mode.
manual_mode = False         # whether to use the manually configure the setting.
verbose_mode = False         # whether to printout more information such as the layerwise parameter.
loadFromTxt = False         # whether to load data from a plain txt file.

is_sort = True
is_dynamic_warp = True
is_data_stream = True
is_dim_slice = True

max_warp_size = 1024
dim_slice_size = [32]
#warp_size_k_li = [11, 12, 13, 14, 15, 16, 17, 18, 19]
warp_size_k_li = [18]
column_slice_size_li = [1]#range(1, 65, 1)
if run_GCN:
    model = 'gcn'
    warpPerBlock = 8  # only effective in manual model
    hidden = [16]
else:
    model = 'gin'
    warpPerBlock = 2  # only effective in manual model 2 for citeseer 6 for remaining datasets
    hidden = [64]

partsize_li = [16]  # only effective in manual model
partsize = 16
dim_cur = 1024 * 2

''' 
		('citeseer'	        		 , 3703	    , 6   , False),  
		('cora' 	        		 , 1433	    , 7   , False),  
		('pubmed'	        		 , 500	    , 3   , False),      
		('ppi'	            		 , 50	    , 121 , False),   

		('PROTEINS_full'             , 29       , 2   , False),   
		('OVCAR-8H'                  , 66       , 2   , False), 
		('Yeast'                     , 74       , 2   , False),
		('DD'                        , 89       , 2   , False),
		('TWITTER-Real-Graph-Partial', 1323     , 2   , False),   
		('SW-620H'                   , 66       , 2   , False),

		( 'amazon0505'               , 96	    , 22  , False),
		( 'artist'                   , 100      , 12  , False),
		( 'com-amazon'               , 96	    , 22  , False),
		( 'soc-BlogCatalog'	         , 128      , 39  , False), 
		( 'amazon0601'  	         , 96	    , 22  , False),
		
        ( 'enwiki-2013'			     , 300      , 12  , True),
        ( 'wiki-topcats'			 , 300      , 12  , True), 
		('ogbn-arxiv'			     , 128      , 10  , True), 
		( 'ogbn-products'			 , 100      , 47  , True),
		( 'ogbn-proteins'			 , 8        , 112 , True), 
'''

python_cmd = common_dataset.python_cmd
for column_slice_size in column_slice_size_li:
    for cur_dim_slice in dim_slice_size:
        for hid in hidden:
            for data, d, c, load,warp_size_k in dataset:
                loadFromTxt = load
                outfile_new = outfile+"-"+data+".txt"
                command = "{} GNNA_main.py --dataset {} --dim {} --hidden {} \
                            --classes {} --partSize {} --model {} --warpPerBlock {}\
                            --manual_mode {} --verbose_mode {} --enable_rabbit {} --loadFromTxt {} --warp_size_k {} --column_slice_size {} --cur_dim_slice {} --max_warp_size {} --is_sort {} --is_dynamic_warp {} --is_data_stream {} --is_dim_slice {} | tee -a {}"
                command = command.format(python_cmd, data, d, hid, c, partsize, model, warpPerBlock, \
                                         manual_mode, verbose_mode, enable_rabbit, loadFromTxt, warp_size_k, column_slice_size, cur_dim_slice, max_warp_size, is_sort, is_dynamic_warp, is_data_stream, is_dim_slice, outfile)
                # command = "python GNNA_main.py -loadFromTxt --dataset {} --partSize {} --dataDir {}".format(data, partsize, '/home/yuke/.graphs/orig')
                os.system(command)
