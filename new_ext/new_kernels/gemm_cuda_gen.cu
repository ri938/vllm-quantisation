/*

@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}

 */


#include <torch/extension.h>
#include "gemm_cuda.h"
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <cstdio>


__global__ void _dequant_slow(
	int* kernel, half* scales, int* zeros, half* out_feats, int in_channels, int num_packed_channels
) {
    /*
      scales IC // 128, OC [float16]
      zeros  IC // 128, OC // 8 [int32]
      kernel IC, OC // 8 [int32]
      output IC, OC
      IC = IF
    */

    int num_output_channels = num_packed_channels * 8;
    int order_map[] = {0, 2, 4, 6, 1, 3, 5, 7}; 

    for (int row = 0; row < in_channels; row++) {
         for (int column = 0; column < num_packed_channels; column++) {
	    for (int pos = 0; pos < 8; pos++) {
		half* s_item = scales + row / 128 * num_output_channels + column * 8 + order_map[pos];
		int* z_item = zeros + row / 128 * num_output_channels / 8 + column;
                int* w_item = kernel + row * num_output_channels / 8 + column;

		half zero = __float2half(static_cast<float>((*z_item >> 4 * pos) & 0xf));
		half weight = __float2half(static_cast<float>((*w_item >> 4 * pos) & 0xf));

		half scaled_zero = __hmul(zero, *s_item);
		half dequant = __hsub(__hmul(weight, *s_item), scaled_zero);

		half* out_ptr = out_feats + row * num_output_channels + column * 8 + order_map[pos];
                *(half*)(out_ptr) = dequant;
            }
         }
    }
}


__global__ void _dequant(
	int* kernel, half* scales, int* zeros, half* out_feats, int in_channels, int num_packed_channels
) {
    /*
      scales IC // 128, OC [float16]
      zeros  IC // 128, OC // 8 [int32]
      kernel IC, OC // 8 [int32]
      output IC, OC
      IC = IF
    */

    int num_output_channels = num_packed_channels * 8;
    int order_map[] = {0, 2, 4, 6, 1, 3, 5, 7}; 

    int idx = blockIdx.x * 512 + threadIdx.y * 32 + threadIdx.x;

    int column = idx % num_packed_channels;
    int row = idx / num_packed_channels;

    if (idx < num_packed_channels * in_channels) {
        int* z_item = zeros + row / 128 * num_output_channels / 8 + column;
        int* w_item = kernel + row * num_output_channels / 8 + column;

        for (int pos = 0; pos < 8; pos++) {
	    half* s_item = scales + row / 128 * num_output_channels + column * 8 + order_map[pos];

	    half zero = __float2half(static_cast<float>((*z_item >> 4 * pos) & 0xf));
	    half weight = __float2half(static_cast<float>((*w_item >> 4 * pos) & 0xf));

	    half scaled_zero = __hmul(zero, *s_item);
	    half dequant = __hsub(__hmul(weight, *s_item), scaled_zero);

	    half* out_ptr = out_feats + row * num_output_channels + column * 8 + order_map[pos];
	    *(half*)(out_ptr) = dequant;
        }
    }
}


// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
torch::Tensor dequantize(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros)
{
    int num_in_feats = _kernel.size(0);
    int num_in_channels = _kernel.size(1) * 8;
    int num_packed_channels = _kernel.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(_kernel));

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(_kernel.device());
    at::Tensor dq = torch::zeros({num_in_feats, num_in_channels}, options);

    if (num_in_feats != _scaling_factors.size(0) * 128)
        throw std::invalid_argument("Invalid scaling factor size:0");
    if (num_in_channels != _scaling_factors.size(1))
        throw std::invalid_argument("Invalid scaling factor size:1");
    if (num_in_feats != _zeros.size(0) * 128)
        throw std::invalid_argument("Invalid zeros size:0");
    if (num_in_channels != _zeros.size(1) * 8)
        throw std::invalid_argument("Invalid zeros size:1");

    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto out_feats = reinterpret_cast<half*>(dq.data_ptr<at::Half>());
    auto scales = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

    dim3 threads(32, 16);
    dim3 blocks((num_in_feats * num_packed_channels + 512 - 1) / 512, 1);

    _dequant<<<blocks, threads>>>(kernel, scales, zeros, out_feats, num_in_feats, num_packed_channels);

    return dq;
}


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

    int order_map[] = {0, 2, 4, 6, 1, 3, 5, 7}; 

    // they have to be the same dimension for this to work
    const int blocksize = 8;

    // preload into shared memory which is 10x faster than load from GMEM
    __shared__ float s_weight[blocksize * blocksize * 8];
    //__shared__ half s_feats[blocksize * blocksize];
    __shared__ half s_feats[blocksize * blocksize];

    // need to make sure there is no repeated x-y pairs
    const int x = blockIdx.x * blocksize + threadIdx.x / blocksize;
    const int y = blockIdx.y * blocksize + threadIdx.x % blocksize;

    // row and column in block-space
    const int chunk_row = x / blocksize;
    const int chunk_column = y / blocksize;

    // position of the thread inside the moving block in either feats / kernel
    const int thread_row = threadIdx.x / blocksize;
    const int thread_column = threadIdx.x % blocksize;

    const int num_output_channels = num_packed_channels * 8;

    // all the blocks start on column 0
    half* in_feats_ptr = in_feats + chunk_row * in_channels * blocksize;

    // all the blocks start on row 0
    int* kernel_ptr = kernel + chunk_column * blocksize;

    if (x < num_in_feats && y < num_packed_channels) {
	// one column of kernel becomes 8 in the output due to dequantising
        float tmp_results[8] = {0.0};

	for (int shift = 0; shift < in_channels; shift+=blocksize) {
           // order not important for correctness but is important for coalescence
           s_feats[thread_row * blocksize + thread_column] = in_feats_ptr[thread_row * in_channels + thread_column];

	   int x_kernel_offset = shift + thread_row;
           int z_item = *(zeros + x_kernel_offset / 128 * num_output_channels / 8 + y);
	   int w_item = kernel_ptr[thread_row * num_packed_channels + thread_column];

	   for (int pos = 0; pos < 8; pos++) {
	       half s_item = *(scales + x_kernel_offset / 128 * num_output_channels + y * 8 + order_map[pos]);

	       float zero = static_cast<float>((z_item >> 4 * pos) & 0xf);
	       float weight = static_cast<float>((w_item >> 4 * pos) & 0xf);

	       float scaled_zero = zero * __half2float(s_item);
	       float dequant = (weight * __half2float(s_item)) - scaled_zero;

	       // index chosen to remove bank conflicts (matrix as pos-row-column order)
               int idx = order_map[pos] * blocksize * blocksize + thread_row * 8 + thread_column;
	       s_weight[idx] = dequant;
           }

	   __syncthreads();

	   // advance the block forward
	   in_feats_ptr += blocksize;
	   kernel_ptr += blocksize * num_packed_channels;

	   // multiply the features and weights together
	   for (int blockpos=0; blockpos < blocksize; blockpos++) {
	       half f_item = s_feats[thread_row * blocksize + blockpos];

	       for (int pos=0; pos < 8; pos++) {
		   float dequant = s_weight[pos * blocksize * blocksize + blockpos * 8 + thread_column];
	           float value = __half2float(f_item) * dequant;
	           tmp_results[pos] +=  value;
	       }
           }

	   // next loop will change the shared memory again
	   __syncthreads();
        }

	// write out the results for this position
        for (int pos=0; pos < 8; pos++) {
	    half* out_ptr = out_feats + x * num_output_channels + y * 8 + pos;
	    *(half*)(out_ptr) = __float2half(tmp_results[pos]);
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

    int num_threads = 8 * 8;
    dim3 threads(num_threads);
    dim3 blocks((num_in_feats + 8 - 1) / 8, (num_packed_channels + 8 - 1) / 8);

    quant_forward_mm<<<blocks, threads>>>(in_feats, kernel, scaling_factors, zeros, out_feats, num_in_channels, num_packed_channels, num_in_feats);

    return _out_feats;
}

