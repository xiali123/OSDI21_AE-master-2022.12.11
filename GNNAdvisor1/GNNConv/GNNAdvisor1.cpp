#include <torch/extension.h>
#include <vector>
#include<stdio.h>
#include<algorithm>
#include<iostream>
#include<algorithm>
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
    int partSize, 
    int dimWorker, 
    int warpPerBlock
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
    int partSize, 
    int dimWorker, 
    int warpPerBlock
  );

std::vector<torch::Tensor> spmm_forward_cuda_gin(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
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
    int partSize, 
    int dimWorker, 
    int warpPerBlock
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
    int partSize, 
    int dimWorker, 
    int warpPerBlock) 
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
                            partSize, dimWorker, warpPerBlock);
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
    int partSize, 
    int dimWorker, 
    int warpPerBlock
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
                            partSize, dimWorker, warpPerBlock);
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
    int partSize, 
    int dimWorker, 
    int warpPerBlock
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
                              partSize, dimWorker, warpPerBlock);
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
    int partSize, 
    int dimWorker, 
    int warpPerBlock
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
                            partSize, dimWorker, warpPerBlock);
}


std::vector<torch::Tensor> build_part(
    int partSize, 
    torch::Tensor indptr
  ) 
{

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
      part2Node[part_counter++] = i;
      if (i == num_nodes - 1 &&  partEnd == indptr_acc[i + 1])
        partPtr[part_counter] = partEnd;
    }
	}
  return {partPtr, part2Node};
}

bool cmp(hashNode a, hashNode b)
{
    return a.num < b.num;
}

std::vector<torch::Tensor> build_part1(
    int partSize,
    int col_slice_size,
    torch::Tensor row_idx,
    torch::Tensor col_idx
)
{
    auto row_index = row_idx.accessor<int, 1>();
    auto col_index = col_idx.accessor<int, 1>();

    int num_nodes = row_idx.size(0) - 1;

    std::vector<int> row_vex;
    std::vector<int> col_vex;
    int vex_map[num_nodes];
    memset(vex_map, 0, sizeof(vex_map));
    int slice_end = (num_nodes+col_slice_size-1)/col_slice_size*col_slice_size;


    for(int i = col_slice_size; i < slice_end; i += col_slice_size)
    {
        for(int j = 0; j < num_nodes; j++)
        {
            int col_beg = row_index[j];
            int col_end = row_index[j+1];
            int cur_col_beg = col_beg + vex_map[j];

            int cur_col_num = 0;
            for(int v = cur_col_beg; v < col_end; v++)
            {
                if(col_index[v] >= i) break;
                cur_col_num++;
                if(cur_col_num >= partSize)
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
        }
    }

    auto partPtr = torch::zeros(col_vex.size()).to(torch::kInt);
    auto part2Node = torch::zeros(row_vex.size()).to(torch::kInt);

    for(int i = 0; i < row_vex.size(); i++)
    {
        part2Node[i] = row_vex[i];
        partPtr[i*2] = col_vex[i*2];
        partPtr[i*2+1] = col_vex[i*2+1];
    }

    return {partPtr, part2Node};
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
    //hash映射
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
    //printf("4.\n");
    return {row_new_table, col_new_table, hash_table};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("SAG", &SAG, "GNNAdvisor base Scatter-and-Gather Kernel (CUDA)");

    m.def("forward", &spmm_forward, "GNNAdvisor forward (CUDA)");
    m.def("backward", &spmm_backward, "GNNAdvisor backward (CUDA)");

    m.def("forward_gin", &spmm_forward_gin, "GNNAdvisor forward GIN (CUDA)");
    m.def("backward_gin", &spmm_backward_gin, "GNNAdvisor forward GIN (CUDA)");

    m.def("build_part", &build_part, "GNNAdvisor backward (CPU)");
    m.def("build_part1", &build_part1, "GNNAdvisor backward (CPU)");
    m.def("build_new_csr", &build_new_csr, "GNNAdvisor1 backward (CPU)");
}