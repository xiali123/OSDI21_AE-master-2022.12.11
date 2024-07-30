#!/bin/bash
dataset_path="/home/hnu/Disk0/xiali/xiali/progress/DCGG/DCGG/dataset/bin/"

dataset=(
#      "citeseer 3703 6 False"
#      "cora 1433 7 False"
#      "pubmed 500 3 False"
#      "ppi 50 121 False"
#      #"PROTEINS_full 29 2 False"
#      "OVCAR-8H 66 2 False"
#      "Yeast 74 2 False"
#      "DD 89 2 False"
#      #"TWITTEl 1323 2 False"
#      "SW-620H 66 2 False"
      "amazon0505 96 22 False"
      "artist 100 12 False"
      "com-amazon 96 22 False"
      "soc-BlogCatalog 128 39 False"
      "amazon0601 96 22 False"
)


is_sort=1
is_dynamic_warp=0
is_data_stream=1
is_dim_slice=1
max_warp_size=1024
column_slice_size_li=(1) #(1 4 8 12 16 32 48 52 60 64 96 128)
dim_slice_li=(4) #(4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 80 96 128)
warp_size_k_li=(11 12 13 14 15 16 17 18 19)
if [[ $1 == 'gcn' ]]; then
    warpPerBlock=8        # only effective in manual model
    hidden=16
else
    warpPerBlock=2        # only effective in manual model 2 for citeseer 6 for remaining datasets
    hidden=64
fi

num_epoches=200
partSize=32

for cur_dim_slice in "${dim_slice_li[@]}"; do
  for warp_size_k in "${warp_size_k_li[@]}"; do
    for column_slice_size in "${column_slice_size_li[@]}"; do
      echo -n "开始 "
      for i in "${dataset[@]}"; do
          IFS=' '
          read -ra ADDR <<< "$i"
          data=${ADDR[0]}
          dim=${ADDR[1]}
          class=${ADDR[2]}
          load=${ADDR[3]}
          echo -n "$data $dim $class $load ${column_slice_size} ${hidden}"
          beg_file="${dataset_path}${data}_beg_pos.bin"
          csr_file="${dataset_path}${data}_csr.bin"
          echo -n "$data, $dim, $class, $load"
          ncu --target-processes all --set detailed -f -o cache_log/${data}_${cur_dim_slice}_${warp_size_k}_${column_slice_size} ./5_aggregation ${beg_file} ${csr_file} ${num_epoches} ${partSize} \
                    ${warpPerBlock} ${cur_dim_slice} ${dim} ${hidden} ${is_sort} ${is_dynamic_warp} \
                    ${is_dim_slice} ${warp_size_k} ${cur_dim_slice} ${max_warp_size} ${column_slice_size}
          #./5_aggregation ${beg_file} ${csr_file} ${num_epoches} ${partSize} \
          #          ${warpPerBlock} ${cur_dim_slice} ${dim} ${hidden} ${is_sort} ${is_dynamic_warp} \
          #          ${is_dim_slice} ${warp_size_k} ${cur_dim_slice} ${max_warp_size} ${column_slice_size}
      done
    done
  done
done

echo -n "结束 "


