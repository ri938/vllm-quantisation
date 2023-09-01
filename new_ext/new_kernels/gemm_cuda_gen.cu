#include <torch/extension.h>
#include "gemm_cuda.h"
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <cstdio>


#define CEIL_DIV(A, B) (((A) + (B)-1) / (B))

// dimensions of moving block for feats-x, kernel-y and common
template<int blocksize_x, int blocksize_y, int blocksize_depth, int tile_y>
__global__ void quant_forward_mm(
	half* in_feats,
       	int* kernel,
       	half* scales,
       	int* zeros,
       	half* out_feats,
       	int in_channels,
       	int num_packed_channels,
	int num_in_feats
) {
    /*
      in_feats M, IC
      scales IC // 128, OC [float16]
      zeros  IC // 128, OC // 8 [int32]
      kernel IC, OC // 8 [int32]
      output IC, OC
      IC = IF
    */

    // AWQ stores 4bits in this order
    int order_map[] = {0, 2, 4, 6, 1, 3, 5, 7}; 

    // one zero and scale per input channel group
    const int groupsize = 128;

    const int num_output_channels = num_packed_channels * 8;

    // preload into shared memory which is 10x faster than load from GMEM
    __shared__ float s_weight[blocksize_y * blocksize_depth * 8];
    __shared__ half s_feats[blocksize_x * blocksize_depth];

    // need to make sure there is no repeated x-y pairs
    const int x = blockIdx.x * blocksize_x + threadIdx.x / blocksize_depth;

    // which entries the thread is responsible for in the feat block
    const int feat_row = threadIdx.x / blocksize_depth;
    const int feat_col = threadIdx.x % blocksize_depth;

    const int num_threads = blockDim.x;

    // each tile is responsible for tile_y consecutive elements so this is just the start
    const int y_offset = blockIdx.y * blocksize_y + (threadIdx.x % tile_y);

    // for the kernel block (in y dimension its the starting position)
    const int kernel_row = (threadIdx.x * tile_y) / blocksize_y;
    const int kernel_col_offset = threadIdx.x % tile_y;
    const int kernel_col_stride = blocksize_y / tile_y;
    //const int kernel_col_stride = blocksize_y / num_threads;

    // position the block pointers at the start
    half* in_feats_ptr = in_feats + blocksize_x * blockIdx.x * in_channels;
    int* kernel_ptr = kernel + blocksize_y * blockIdx.y;

    // one column of kernel becomes 8 in the output due to dequantising
    // and each thread is responsible for tile-y elements in the kernel
    float tmp_results[8 * tile_y] = {0.0};

    for (int shift = 0; shift < in_channels; shift+=blocksize_depth) {
       // order not important for correctness but is important for coalescence
       // relies on assumption that num threads == size of feats block
       s_feats[feat_row * blocksize_depth + feat_col] = in_feats_ptr[feat_row * in_channels + feat_col];

       // load in column order to reduce bank conflicts
       for (int tid=0; tid < tile_y; tid++) {
           //int x_kernel_offset = shift + kernel_row;
	   int x_block_offset = tid;
	   int y_block_offset = threadIdx.x;

           int x_kernel_offset = shift + tid;
           int y_kernel_offset = blockIdx.y * blocksize_y + threadIdx.x;
           //int y_kernel_offset = y_offset + tid * kernel_col_stride;
           //int y_kernel_offset = y_offset + threadIdx.x;

           int z_item = *(zeros + x_kernel_offset / groupsize * num_output_channels / 8 + y_kernel_offset);
           //int w_item = kernel_ptr[kernel_row * num_packed_channels + kernel_col_offset + tid * kernel_col_stride];
           int w_item = kernel_ptr[x_block_offset * num_packed_channels + y_block_offset];

           // calculate and store the dequantized weights
           for (int pos = 0; pos < 8; pos++) {
               half s_item = *(scales + x_kernel_offset / groupsize * num_output_channels + y_kernel_offset * 8 + order_map[pos]);

               float zero = static_cast<float>((z_item >> 4 * pos) & 0xf);
               float weight = static_cast<float>((w_item >> 4 * pos) & 0xf);

               float scaled_zero = zero * __half2float(s_item);
               float dequant = (weight * __half2float(s_item)) - scaled_zero;

               // index chosen to remove bank conflicts (matrix as pos-row-column order)
               //int idx = order_map[pos] * blocksize_depth * blocksize_y + kernel_row * blocksize_depth + kernel_col_offset + tid;
               //int idx = order_map[pos] * blocksize_depth * blocksize_y + kernel_row * blocksize_y + kernel_col_offset + tid * kernel_col_stride;
               int idx = order_map[pos] * blocksize_depth * blocksize_y + x_block_offset * blocksize_y + y_block_offset;
               s_weight[idx] = dequant;
           }
       }

       __syncthreads();

       // advance the block forward
       in_feats_ptr += blocksize_depth;
       kernel_ptr += blocksize_depth * num_packed_channels;

       // multiply the features and weights together
       for (int blockpos=0; blockpos < blocksize_depth; blockpos++) {
           half f_item = s_feats[feat_row * blocksize_depth + blockpos];

           for (int tid=0; tid < tile_y; tid++) {
               for (int pos=0; pos < 8; pos++) {
	           //float dequant = s_weight[pos * blocksize_depth * blocksize_y + blockpos * blocksize_depth + kernel_col_offset + tid];
	           float dequant = s_weight[pos * blocksize_depth * blocksize_y + blockpos * blocksize_y + kernel_col_offset + tid * kernel_col_stride];
	           float value = __half2float(f_item) * dequant;
	           tmp_results[tid * 8 + pos] +=  value;
               }
           }
       }

       // next loop will change the shared memory again
       __syncthreads();
    }

    // write out the results for this position
    for (int pos=0; pos < 8; pos++) {
	for (int tid=0; tid < tile_y; tid++) {
            half* out_ptr = out_feats + x * num_output_channels + (y_offset + tid * kernel_col_stride) * 8 + pos;
            *(half*)(out_ptr) = __float2half(tmp_results[tid * 8 + pos]);
	}
    }

}

// feats: M, IC
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// out: M, OC
torch::Tensor gemm_forward_cuda_new(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    int num_out_channels = _kernel.size(1) * 8;
    int num_packed_channels = _kernel.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());

    at::Tensor _out_feats = torch::zeros({num_in_feats, num_out_channels}, options);

    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

    if (_in_feats.size(1) != _kernel.size(0))
        throw std::invalid_argument("Kernel and input channels mismatch dim0");
    if (_scaling_factors.size(1) != _kernel.size(1) * 8)
        throw std::invalid_argument("Kernel and scaling factors mismatch dim1");

    if (num_in_channels != _scaling_factors.size(0) * 128)
        throw std::invalid_argument("Invalid scaling factors size (dim1)");
    if (num_out_channels != _scaling_factors.size(1))
        throw std::invalid_argument("Invalid scaling factors size (dim1)");

    if (_zeros.size(0) != _scaling_factors.size(0))
        throw std::invalid_argument("Invalid zeros size (dim0)");
    if (_zeros.size(1) * 8 != num_out_channels)
        throw std::invalid_argument("Invalid zeros size (dim1)");

    if (num_in_feats % 8 != 0) {
        throw std::invalid_argument("In feats must be divisible by 8");
    }

    if (num_packed_channels % 8 != 0) {
        throw std::invalid_argument("Packed channels must be divisible by 8");
    }

    const int blocksize_y = 64;
    const int blocksize_x = 8;
    const int blocksize_depth = 8;
    const int tiles_y = 8;

    assert(num_in_feats % blocksize_x == 0);
    assert(num_in_channels % blocksize_depth == 0);
    assert(num_packed_channels % blocksize_y == 0);

    // each thread will be responsible for tiles_y channels in the y-dimension
    const int num_threads = blocksize_x * blocksize_y / tiles_y;
    dim3 threads(num_threads);

    // check the shapes compatible
    assert(blocksize_y % tiles_y == 0);

    // check that threads can load the data as expected
    assert(num_threads == blocksize_x * blocksize_depth);
    assert(num_threads == blocksize_y * blocksize_depth / tiles_y);

    dim3 blocks(CEIL_DIV(num_in_feats, blocksize_x), CEIL_DIV(num_packed_channels, blocksize_y));

    quant_forward_mm<blocksize_x, blocksize_y, blocksize_depth, tiles_y><<<blocks, threads>>>(
        in_feats, kernel, scaling_factors, zeros, out_feats, num_in_channels,
       	num_packed_channels, num_in_feats
    );

    return _out_feats;
}

