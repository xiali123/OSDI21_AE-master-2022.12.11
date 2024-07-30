#!/usr/bin/env python3
import os
import time
import sys
sys.path.append('../')
import common_dataset
os.environ["PYTHONWARNINGS"] = "ignore"

dataset=common_dataset.dataset


islocal = common_dataset.islocal
islocal=True
if islocal:
    dataset = [
        # ('citeseer'	        ,3703	    , 6   , False, 17),
        # ('cora' 	        		 , 1433	    , 7   ,False, 17),
        # ('pubmed'	        		 ,500	    , 3   , False, 17),
        # ('ppi'	            		 , 50	    ,121 , False, 17),
        #
        # ('PROTEINS_full'             ,29       , 2   , False, 17),
        # ('OVCAR-8H'                  , 66       , 2   , False, 17),
        # ('Yeast'                     , 74       , 2   , False, 17),
        # ('DD'                        , 89       , 2   , False, 17),
        # ('TWITTER-Real-Graph-Partial', 1323     , 2   , False, 17),
        # ('SW-620H'                   , 66       ,2    , False, 17),

        ( 'amazon0505'               , 96	    ,22  , False, 17),
        ( 'artist'                   , 100, 12  , False, 17),
        ( 'com-amazon'               , 96	    , 22  , False, 17),
        ('soc-BlogCatalog'	         , 128, 39  , False, 17),
        ( 'amazon0601'  	         , 96	    , 22  , False, 17),
    ]

mytime = str(time.time())
outfile = './log/4_hidden_acc_GCN_' + mytime  +'.log'

run_GCN = True              # whether to run GCN model.
enable_rabbit = False        # whether to enable rabbit reordering in auto and manual mode.
manual_mode = False         # whether to use the manually configure the setting.
verbose_mode = False         # whether to printout more information such as the layerwise parameter.
loadFromTxt = False         # whether to load data from a plain txt file.

max_warp_size = 1024

if run_GCN:
    model = 'gcn'
    warpPerBlock = 8        # only effective in manual model
    hidden = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 48, 64, 80, 96, 112, 128, 256, 384, 512, 1024, 2048, 4096]
else:
    model = 'gin'
    warpPerBlock = 2        # only effective in manual model 2 for citeseer 6 for remaining datasets
    hidden = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 48, 64, 80, 96, 112, 128, 256, 384, 512, 1024, 2048, 4096]

partsize_li = [16]          # only effective in manual model
partsize = 16

python_cmd = common_dataset.python_cmd
for hid in hidden:
    for data, d, c, load, warp_size_k in dataset:
        loadFromTxt = load
        outfile_new = outfile+"-"+data+".txt"
        command = "{} GNNA_main.py --dataset {} --dim {} --hidden {} \
                    --classes {} --partSize {} --model {} --warpPerBlock {}\
                    --manual_mode {} --verbose_mode {} --enable_rabbit {} --loadFromTxt {}  --warp_size_k {} --max_warp_size {} | tee -a {}"
        command = command.format(python_cmd, data, d, hid, c, partsize, model, warpPerBlock, \
                                 manual_mode, verbose_mode, enable_rabbit, loadFromTxt, warp_size_k, max_warp_size, outfile)
        # command = "python GNNA_main.py -loadFromTxt --dataset {} --partSize {} --dataDir {}".format(data, partsize, '/home/yuke/.graphs/orig')
        os.system(command)
