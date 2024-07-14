#include <vector>
#include<stdio.h>
#include<algorithm>
#include<iostream>
#include<algorithm>
#include <cuda.h>
#include <cublas_v2.h>
#include "graph.h"

using nidType = int;
// using nidType = long;

using namespace std;

#define max(a,b) (a>b)?a:b
#define min(a,b) (a>b)?b:a
#define WARP_SIZE 32
typedef struct hashNode
{
    int index;
    int num;
}hashNode;

__device__ inline
void atomicAdd_F(float* address, float value)
{
  float old = value;
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
}

bool cmp(hashNode a, hashNode b)
{
    return a.num < b.num;
}

int getPartSize_inc(
    int limit_partsize,
    int degree,
    int old_degree,
    int avg_degree,
    int cur_part,
    double k
)
{
    double var_degree = (double)degree;
    double res = log2(var_degree);

    int t = (int)(pow(k/10, res));

    return min((max(t, avg_degree)), limit_partsize);
}

std::vector<std::vector<int>> build_part1(
    int partSize,
    int col_slice_width,
    int max_degree,
    double k,
    int limit_dgre,
    std::vector<int> &row_index,
    std::vector<int> &col_index
)
{
    int num_nodes = row_index.size() - 1;

    std::vector<int> row_vex;
    std::vector<int> col_vex;
    int vex_map[num_nodes];
    memset(vex_map, 0, sizeof(vex_map));
    int col_slice_size = (num_nodes+col_slice_width-1)/col_slice_width;
    int slice_end = col_slice_size*col_slice_width;

    int part_size = partSize;
    int max_part = partSize;
    int old_degree = 1;

    int limit_degree = max_degree;
    if(max_degree > 256) limit_degree = min(limit_dgre, max_degree/2);
    part_size = min(limit_degree, part_size);

    for(int i = col_slice_size; i <= slice_end; i += col_slice_size)
    {
        for(int j = 0; j < num_nodes; j++)
        {
            int col_beg = row_index[j];
            int col_end = row_index[j+1];
            int cur_col_beg = col_beg + vex_map[j];
            int degree = col_end - col_beg;

            if(max_degree < 32) part_size = max_degree;
            else if(degree <= partSize) part_size = partSize;
            else part_size = max(partSize, getPartSize_inc(limit_degree, degree, old_degree, partSize, part_size, k));

            max_part = max(part_size, max_part);
            int cur_col_num = 0;
            for(int v = cur_col_beg; v < col_end; v++)
            {
                if(col_index[v] >= i) break;
                cur_col_num++;
                if(cur_col_num >= part_size)
                {
                    vex_map[j] += cur_col_num;
                    row_vex.push_back(j);
                    col_vex.push_back(cur_col_beg);
                    col_vex.push_back(col_beg+vex_map[j]);
                    cur_col_beg = col_beg + vex_map[j];
                    cur_col_num = 0;
                }
            }

            vex_map[j] += cur_col_num;
            if(cur_col_num > 0)
            {
                row_vex.push_back(j);
                col_vex.push_back(cur_col_beg);
                col_vex.push_back(col_beg+vex_map[j]);
            }

            old_degree = degree;
        }
    }

    std::vector<int> partPtr;
    std::vector<int> part2Node;
    std::vector<int> partInfo;
    for(int i = 0; i < row_vex.size(); i++)
    {
        part2Node.push_back(row_vex[i]);
        partPtr.push_back(col_vex[i*2]);
        partPtr.push_back(col_vex[i*2+1]);
    }
    partInfo.push_back(max_part);
    return {partPtr, part2Node, partInfo};
}

std::vector<std::vector<int>> build_new_csr(
    std::vector<int> degrees,                  //就是度数
    std::vector<int> row_table,
    std::vector<int> column_table
)
{
    int num_vexs = degrees.size();
    //hash映射
    //printf("1.\n");
    auto degrees_ptr = degrees;
    auto row_pointer = row_table;
    auto column_pointer = column_table;

        //hash映射
    //printf("2.%d\n", num_vexs);
    std::vector<int> hash_table_ptr(num_vexs, 0);
    std::vector<hashNode> hash_vct;
    hashNode tag_hash;
    //printf("2.----->%d\n", num_vexs);
    for(int i = 0; i < num_vexs; i++)
    {
        //printf("degrees_ptr[%d]: %d\n",i, degrees_ptr[i]);
        tag_hash.index = i;
        tag_hash.num = degrees_ptr[i];
        hash_vct.push_back(tag_hash);
    }
    std::sort(hash_vct.begin(), hash_vct.end(), cmp);

    std::vector<int> new_degree_ptr(degrees.size());
    std::vector<int> row_new_ptr(row_table.size());
    std::vector<int> col_new_ptr(column_table.size());
    //hash映射
    //printf("3.\n");
    //新的csr结构
    int c = 0;
    row_new_ptr[0] = 0;
    for(int i = 1; i <= hash_vct.size(); i++)
    {
        int hash_tag = hash_vct[i-1].index;
        int cur_degree_num = hash_vct[i-1].num;
        int col_pos = row_pointer[hash_tag];
        hash_table_ptr[i-1] = hash_tag;
        new_degree_ptr[i-1] = cur_degree_num;
        row_new_ptr[i] = row_new_ptr[i-1] + cur_degree_num;
        for(int j = 0; j < cur_degree_num; j++) col_new_ptr[c++] = column_pointer[col_pos+j];
    }
        //hash映射
    //printf("4.\n");
    return {row_new_ptr, col_new_ptr, new_degree_ptr};
}



__global__ void spmm_forward_cuda_kernel_gin(
    float * output,
    float * input,
    int * row_pointers,
    int * column_index,
    float epsilon,
    int * part_pointers,
    int * part2Node,
    const int num_nodes,
    const int dim,
    const int num_parts,
    const int total_num_parts,
    const int dim_per_part,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
);

void spmm_forward_cuda_gin(
    float * output,
    float * input,
    int * row_pointers,
    int * column_index,
    float epsilon,
    int * part_pointers,
    int * part2Node,
    int dim,
    int num_nodes,
    int num_parts,
    int partSize, 
    int dimWorker, 
    int warpPerBlock,
    int dim_per_part
) 
{
    dim_per_part = min(dim, dim_per_part);
    const int total_num_parts = (dim + dim_per_part-1)/dim_per_part *num_parts;

    const int block = min(warpPerBlock*WARP_SIZE, 1024);
    const int grid = (total_num_parts*WARP_SIZE + block - 1) / block;
    const int shared_memory = warpPerBlock*partSize*sizeof(int) + warpPerBlock*dim_per_part*sizeof(float);

    spmm_forward_cuda_kernel_gin<<<grid, block, shared_memory>>>(output, input, row_pointers, column_index, epsilon, part_pointers, part2Node, num_nodes, dim, num_parts, total_num_parts, dim_per_part, partSize, dimWorker, warpPerBlock);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return ;
}


__global__ void spmm_forward_cuda_kernel_gin(
    float * output,
    float * input,
    int * row_pointers, 
    int * column_index,
    float epsilon,
    int * part_pointers,
    int * part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int total_num_parts,
    const int dim_per_part,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
) 
{

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / WARP_SIZE;                             // global warp-id
    int block_warpId = threadIdx.x / WARP_SIZE;               // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid
    int cur_dim_base = warpId / num_parts * dim_per_part;
    int cur_warp_id = warpId % num_parts;
    int cur_dim_size = (dim_per_part > dim-cur_dim_base)? dim-cur_dim_base: dim_per_part;
    warpId = total_num_parts - warpId - 1;
    extern __shared__ int part_meta[];                                      // part information.
    int *partial_ids = part_meta;                                           // caching ids
    float *partial_results = (float*)&part_meta[partSize*warpPerBlock];     // caching partial results.

    if (warpId >= 0){

        int srcId = part2Node[cur_warp_id];              // aggregated source node
        //int partBeg = part_pointers[warpId];        // partitioning pointer start
        //int partEnd = part_pointers[warpId + 1];    // part pointer end
        const int partBeg = part_pointers[cur_warp_id*2];
        const int partEnd = part_pointers[cur_warp_id*2 + 1];
        // Cache the part neighbors.
        const int pindex_base = block_warpId * partSize;
        #pragma unroll
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += dimWorker){
            partial_ids[pindex_base + nidx - partBeg] = column_index[nidx];
        }

         __syncwarp();

        // Neighbor aggregation within each part
        const int presult_base = block_warpId * dim_per_part;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];

            // Initialize shared memory for partial results
            if (nIdx == 0)
                if (laneid < dimWorker)
                #pragma unroll
                for (int d = laneid; d < cur_dim_size; d += dimWorker){
                    partial_results[presult_base + d] = 0.0f;
                }
            
            if (laneid < dimWorker)
            #pragma unroll
            for (int d = laneid; d < cur_dim_size; d += dimWorker){
                partial_results[presult_base + d] += input[nid*dim+d+cur_dim_base];
            }
        }

        // output the result to global memory from the shared memory
        if (laneid < dimWorker)
        #pragma unroll
        for (int d = laneid; d < cur_dim_size; d += dimWorker){
            atomicAdd_F((float*)&output[srcId*dim+d+cur_dim_base], epsilon*partial_results[presult_base + d]);
        }
    }
}


int main(int argc, char *argv[])
{
    //导入参数
    if (argc < 8){
        printf("Usage: ./main graph.mtx num_GPUs partSize warpPerblock dim interleaved_dist hidden\n");
        return -1;
    }

    cout << "Graph File: " << argv[1] << '\n';
    const char *beg_file = argv[1];
	const char *csr_file = argv[2];

	//printf("point 0 \n");
    int num_epoches = atoi(argv[3]);           // 2
    int partSize = atoi(argv[4]);           // 32
    int warpPerBlock = atoi(argv[5]);       // 4
    int dim_per_part = atoi(argv[6]);   // 2
    int dim = atoi(argv[7]);                // 16
    int hiddenSize = atoi(argv[8]);
    int outdim = hiddenSize;
    int is_sort = atoi(argv[9]);
    int is_dynamic_warp = atoi(argv[10]);
    int is_dim_slice = atoi(argv[11]);
    int warp_size_k = atoi(argv[12]);

    int cur_dim_slice = atoi(argv[13]);
    int max_warp_size = atoi(argv[14]);
    int column_slice_size = atoi(argv[15]);
    int dimWorker = 32;


    //printf("point 1 \n");
    //导入图像
    graph<long, long, nidType, nidType, nidType, nidType>* ginst = new graph<long, long, nidType, nidType, nidType, nidType>(beg_file, csr_file);
    std::vector<nidType> row_pointers(ginst->beg_pos, ginst->beg_pos + ginst->vert_count + 1);
    std::vector<nidType> column_index(ginst->csr, ginst->csr + ginst->edge_count);

    int num_nodes = row_pointers.size() -1;
    int num_edges = column_index.size();
    //预处理
    std::vector<nidType> degreeTable;
    for(int i = 1; i < row_pointers.size(); i++) degreeTable.push_back(row_pointers[i]-row_pointers[i-1]);
    //printf("point 2 \n");

    auto split_output = build_new_csr(degreeTable, row_pointers, column_index);
    auto new_row_pointers = split_output[0];
    auto new_col_pointers = split_output[1];
    auto new_degree_ptr = split_output[2];


    std::vector<int> &res_row_pointers = row_pointers;
    std::vector<int> &res_col_pointers = column_index;
    std::vector<int> &res_degree_ptr = degreeTable;


    if (is_sort)
    {
        res_row_pointers = new_row_pointers;
        res_col_pointers = new_col_pointers;
        res_degree_ptr = new_degree_ptr;
    }
    int max_degree = new_degree_ptr[new_degree_ptr.size()-1];
    auto split_output1 = build_part1(partSize, int(column_slice_size),int(max_degree), warp_size_k, max_warp_size, res_row_pointers, res_col_pointers);
    auto new_partPtr = split_output1[0];
    auto new_part2Node = split_output1[1];
    auto new_partInfo = split_output1[2];


    //创建数据
    //printf("point 3 \n");
    int num_parts = new_part2Node.size();
    int *part2Node_d, *partPtr_d, *d_row_ptr_l, *d_col_ind_l;
    cudaMalloc((void**)&d_row_ptr_l, res_row_pointers.size()*sizeof(int));
    cudaMalloc((void**)&d_col_ind_l, res_col_pointers.size()*sizeof(int));
    cudaMemcpy(d_row_ptr_l, &res_row_pointers[0], res_row_pointers.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind_l, &res_col_pointers[0], res_col_pointers.size()*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&partPtr_d, new_partPtr.size()*sizeof(int));
    cudaMalloc((void**)&part2Node_d, new_part2Node.size()*sizeof(int));
    cudaMemcpy(partPtr_d, &new_partPtr[0], new_partPtr.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(part2Node_d, &new_part2Node[0], new_part2Node.size()*sizeof(int), cudaMemcpyHostToDevice);
    //printf("point 3.5 \n");

    float *output, *input, *h_input;

    h_input = (float *)malloc(dim*num_nodes*sizeof(float));
    memset(h_input, 1, sizeof(h_input));
    cudaMalloc((void**)&output, dim*num_nodes*sizeof(float));
    cudaMalloc((void**)&input, dim*num_nodes*sizeof(float));

    cudaMemcpy(input, h_input, dim*num_nodes*sizeof(float), cudaMemcpyHostToDevice);
    //训练
    //printf("point 4 \n");

    float epsilon = 0.5;
    //cudaEvent_t e_start, e_end;
    //cudaEventCreate(&e_start);
    //cudaEventCreate(&e_end);
    //cudaEventRecord(e_start);
    for(int i = 0; i < 1; i++)
    {
        spmm_forward_cuda_gin(output, input, d_row_ptr_l, d_col_ind_l, epsilon, partPtr_d, part2Node_d, dim, num_nodes, num_parts, partSize, dimWorker, warpPerBlock, dim_per_part);
    }
    cudaDeviceSynchronize();
    //cudaEventRecord(e_end);
    //cudaEventSynchronize(e_end);
    //float elapse_time = 0.0;
    //cudaEventElapsedTime(&elapse_time, e_start, e_end);
    //cudaEventDestroy(e_start);
    //cudaEventDestroy(e_end);
    //printf("Preproc (ms): %.3f\n", elapse_time);
    return 1;
}