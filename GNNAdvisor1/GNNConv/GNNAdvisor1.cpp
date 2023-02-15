#include <torch/extension.h>
#include <vector>
#include<stdio.h>
#include<algorithm>
#include<iostream>
#include<algorithm>

#define max(a,b) (a>b)?a:b;
#define min(a,b) (a>b)?b:a;
typedef struct hashNode
{
    int index;
    int num;
}hashNode;


torch::Tensor SAG_cuda(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
);


std::vector<torch::Tensor> spmm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    //torch::Tensor block_hash,
    int partSize, 
    int dimWorker, 
    int warpPerBlock,
    int smSize
);

std::vector<torch::Tensor> spmm_backward_cuda(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    //torch::Tensor block_hash,
    int partSize, 
    int dimWorker, 
    int warpPerBlock,
    int smSize
  );

std::vector<torch::Tensor> spmm_forward_cuda_gin(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    //torch::Tensor block_hash,
    int partSize, 
    int dimWorker, 
    int warpPerBlock,
    int smSize
  );

std::vector<torch::Tensor> spmm_backward_cuda_gin(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    //torch::Tensor block_hash,
    int partSize, 
    int dimWorker, 
    int warpPerBlock,
    int smSize
  );

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor SAG(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index, 
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock) 
{
  CHECK_INPUT(input);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(degrees);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return SAG_cuda(input, row_pointers, column_index, 
              degrees, part_pointers, part2Node, 
              partSize, dimWorker, warpPerBlock);
}


std::vector<torch::Tensor> spmm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index, 
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    //torch::Tensor block_hash,
    int partSize, 
    int dimWorker, 
    int warpPerBlock,
    int smSize
)
{
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(degrees);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_forward_cuda(input, weight, row_pointers, column_index, 
                            degrees, part_pointers, part2Node,
                            partSize, dimWorker, warpPerBlock, smSize);
}

std::vector<torch::Tensor> spmm_backward(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    //torch::Tensor block_hash,
    int partSize, 
    int dimWorker, 
    int warpPerBlock,
    int smSize
  ) 
{

  CHECK_INPUT(d_output);
  CHECK_INPUT(X);
  CHECK_INPUT(W);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(degrees);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_backward_cuda(d_output, X, W, row_pointers, column_index, 
                            degrees, part_pointers, part2Node,
                            partSize, dimWorker, warpPerBlock, smSize);
}


////////////////////////////////
// spmm forward GIN
///////////////////////////////
std::vector<torch::Tensor> spmm_forward_gin(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index, 
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    //torch::Tensor block_hash,
    int partSize, 
    int dimWorker, 
    int warpPerBlock,
    int smSize
  ) 
{
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_forward_cuda_gin(input, weight, row_pointers, column_index, 
                              epsilon, part_pointers, part2Node,
                              partSize, dimWorker, warpPerBlock, smSize);
}

////////////////////////////////
// spmm backward GIN
///////////////////////////////
std::vector<torch::Tensor> spmm_backward_gin(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    //torch::Tensor block_hash,
    int partSize, 
    int dimWorker, 
    int warpPerBlock,
    int smSize
  )
{
  CHECK_INPUT(d_output);
  CHECK_INPUT(X);
  CHECK_INPUT(W);
  CHECK_INPUT(row_pointers);
  CHECK_INPUT(column_index);
  CHECK_INPUT(part_pointers);
  CHECK_INPUT(part2Node);

  return spmm_backward_cuda_gin(d_output, X, W, row_pointers, column_index, 
                            epsilon, part_pointers, part2Node,
                            partSize, dimWorker, warpPerBlock, smSize);
}

bool cmp(hashNode a, hashNode b)
{
    return a.num < b.num;
}

torch::Tensor build_block_hash(
    int num_parts,
    int smSize,
    int block,
    int warpsize
)
{
    int block_total_size = (num_parts * warpsize + block  - 1) / block;
    torch::Tensor block_hash_table = torch::zeros(block_total_size).to(torch::kInt);
    auto block_hash_ptr = block_hash_table.accessor<int, 1>();
    int Block_Num = smSize * smSize;
    for(int i = 0; i < block_total_size; i++)
    {
        int Loop_grid_warpId = i / Block_Num;
        int Loop_grid_lainId = i % Block_Num;

        int L_warpId = Loop_grid_lainId / smSize;
        int L_lainId = Loop_grid_lainId % smSize;

        int cur_max_size = (Loop_grid_warpId+1)*Block_Num > block_total_size? block_total_size:(Loop_grid_warpId+1)*Block_Num;
        int cur_Loop_grid_size = cur_max_size - Loop_grid_warpId * Block_Num;

        int Row_Max_Size = cur_Loop_grid_size / smSize;
        int Col_Max_Size = cur_max_size % smSize;
        if(Loop_grid_lainId < Col_Max_Size) Row_Max_Size++;
        int Hash_warpId = L_warpId;
        if(Row_Max_Size > 0)
        {
            Hash_warpId = (L_warpId - L_lainId % Row_Max_Size + Row_Max_Size) % Row_Max_Size;
        }
        int Hash_lainId = L_lainId;
        block_hash_ptr[i] = Loop_grid_warpId * Block_Num + Hash_warpId * smSize + Hash_lainId;
    }
    return block_hash_table;
}

std::vector<torch::Tensor> build_new_csr(
    torch::Tensor degrees,                  //就是度数
    torch::Tensor row_table,
    torch::Tensor column_table
)
{
    int num_vexs = degrees.size(0);
    //hash映射
    //printf("1.\n");
    auto degrees_ptr = degrees.accessor<int, 1>();
    auto row_pointer = row_table.accessor<int, 1>();
    auto column_pointer = column_table.accessor<int, 1>();

    //hash映射
    //printf("2.%d\n", num_vexs);
    torch::Tensor hash_table = torch::zeros_like(degrees);
    auto hash_table_ptr = hash_table.accessor<int, 1>();
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
    //h
    //printf("3.\n");
    //新的csr结构
    torch::Tensor row_new_table = torch::zeros_like(row_table).to(torch::kInt);
    torch::Tensor col_new_table = torch::zeros_like(column_table).to(torch::kInt);
    auto row_new_ptr = row_new_table.accessor<int, 1>();
    auto col_new_ptr = col_new_table.accessor<int, 1>();
    int c = 0;
    row_new_ptr[0] = 0;
    for(int i = 1; i <= hash_vct.size(); i++)
    {
        int hash_tag = hash_vct[i-1].index;
        int cur_degree_num = hash_vct[i-1].num;
        int col_pos = row_pointer[hash_tag];
        hash_table_ptr[i-1] = hash_tag;
        row_new_ptr[i] = row_new_ptr[i-1] + cur_degree_num;
        for(int j = 0; j < cur_degree_num; j++) col_new_ptr[c++] = column_pointer[col_pos+j];
    }

    //hash映射
    torch::Tensor col_degree_table = torch::zeros_like(degrees).to(torch::kInt);
    auto col_degree_ptr = col_degree_table.accessor<int, 1>();
    for(int i = 0; i < num_vexs; i++)
    {
        col_degree_ptr[i] = degrees_ptr[hash_table_ptr[i]];
    }
    //printf("4.\n");
    return {row_new_table, col_new_table, col_degree_table, hash_table};
}


int getPartSize_inc(
    int limit_partsize,
    int degree,
    int old_degree,
    int cur_part,
    double k
)
{
    double var_degree = (double)degree;
    double var_old_degree = (double)old_degree;
    //int t = cur_part + (int)((var_degree-var_old_degree)/var_old_degree/k * cur_part);
    int t = (int)((var_degree)/k/cur_part*cur_part);
    return min(t, limit_partsize);
}

std::vector<torch::Tensor> build_part1(
    int partSize, 
    int max_degree,
    double k,
    torch::Tensor indptr,
    torch::Tensor hash_table
)
{
    //printf("build_part:%d %d\n", indptr.size(0), hash_table.size(0));
    auto hash_table_ptr = hash_table.accessor<int, 1>();
    auto indptr_acc = indptr.accessor<int, 1>();
    int num_nodes = indptr.size(0) - 1;
    int degree, thisNumParts, numParts = 0;
    int part_size = partSize;
    int old_degree = 1;
    int max_part = partSize;

    int limit_degree = max_degree;
    if(max_degree > 256) limit_degree = min(256, max_degree/2);
    for(int i = 0; i < num_nodes; i++)
    {
        int src_idx = hash_table_ptr[i];
        degree = indptr_acc[i + 1] - indptr_acc[i];
        //sum += degree;
        if(max_degree < 32) part_size = max_degree;
        else if(degree <= partSize) part_size = partSize;
        else part_size = max(partSize, getPartSize_inc(limit_degree, degree, old_degree, partSize, k));

        max_part = max(part_size, max_part)
        if(degree % part_size == 0)
            thisNumParts = degree / part_size;
        else
            thisNumParts = degree / part_size + 1;
        numParts += thisNumParts;
        old_degree = degree;
    }

    auto partPtr = torch::zeros(numParts + 1);
    auto part2Node = torch::zeros(numParts);
    torch::Tensor partInfo = torch::zeros(1).to(torch::kInt);
    auto partInfo_ptr = partInfo.accessor<int, 1>();
    part_size = partSize;
    int part_counter = 0;
    for(int i = 0, old_degree = 0; i < num_nodes; i++)
    {
        int src_idx = hash_table_ptr[i];
        degree = indptr_acc[i + 1] - indptr_acc[i];

        if(max_degree < 32) part_size = max_degree;
        else if(degree <= partSize) part_size = partSize;
        else part_size = part_size = max(partSize, getPartSize_inc(limit_degree, degree, old_degree, partSize, k));

        if(degree % part_size == 0)
            thisNumParts = degree / part_size;
        else
            thisNumParts = degree / part_size + 1;

        for (int pid = 0; pid < thisNumParts; pid++){
            int partBeg = indptr_acc[i] + pid * part_size;
            int partEnd = partBeg + part_size < indptr_acc[i + 1]? partBeg + part_size: indptr_acc[i + 1];
            partPtr[part_counter] = partBeg;
            part2Node[part_counter++] = src_idx;                                            //映射，修改的地方
            if (i == num_nodes - 1 &&  partEnd == indptr_acc[i + 1])
                partPtr[part_counter] = partEnd;
        }
        old_degree = degree;
    }

    partInfo_ptr[0] = max_part;
    return {partPtr, part2Node, partInfo};
}


std::vector<torch::Tensor> build_part(
    int partSize,
    torch::Tensor indptr
  )
{
    //printf("build_part:%d %d\n", indptr.size(0), hash_table.size(0));
  auto indptr_acc = indptr.accessor<int, 1>();
  int num_nodes = indptr.size(0) - 1;
  int degree, thisNumParts, numParts = 0;

	for(int i = 0; i < num_nodes; i++)
	{
    degree = indptr_acc[i + 1] - indptr_acc[i];
	  if(degree % partSize == 0)
			thisNumParts = degree / partSize;
    else
			thisNumParts = degree / partSize + 1;
    numParts += thisNumParts;
	}

  auto partPtr = torch::zeros(numParts + 1);
  auto part2Node = torch::zeros(numParts);

  int part_counter = 0;
	for(int i = 0; i < num_nodes; i++)
	{
    int degree = indptr_acc[i + 1] - indptr_acc[i];
    if(degree % partSize == 0)
			thisNumParts = degree / partSize ;
    else
			thisNumParts = degree / partSize + 1;

    for (int pid = 0; pid < thisNumParts; pid++){
      int partBeg = indptr_acc[i] + pid * partSize;
      int partEnd = partBeg + partSize < indptr_acc[i  + 1]? partBeg + partSize: indptr_acc[i + 1];
      partPtr[part_counter] = partBeg;
      part2Node[part_counter++] = i;                                            //映射，修改的地方
      if (i == num_nodes - 1 &&  partEnd == indptr_acc[i + 1])
        partPtr[part_counter] = partEnd;
    }
	}
  return {partPtr, part2Node};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("SAG", &SAG, "GNNAdvisor1 base Scatter-and-Gather Kernel (CUDA)");

  m.def("forward", &spmm_forward, "GNNAdvisor1 forward (CUDA)");
  m.def("backward", &spmm_backward, "GNNAdvisor1 backward (CUDA)");

  m.def("forward_gin", &spmm_forward_gin, "GNNAdvisor1 forward GIN (CUDA)");
  m.def("backward_gin", &spmm_backward_gin, "GNNAdvisor1 forward GIN (CUDA)");

  m.def("build_part", &build_part, "GNNAdvisor1 backward (CPU)");
  m.def("build_new_csr", &build_new_csr, "GNNAdvisor1 backward (CPU)");
  m.def("build_part1", &build_part1, "GNNAdvisor1 backward (CPU)");
  m.def("build_block_hash", &build_block_hash, "GNNAdvisor1 backward (CPU)");
  }