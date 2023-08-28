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


__global__ void _quant_mm(
	half* in_feats,
       	int* kernel,
       	half* scales,
       	int* zeros,
       	half* out_feats,
       	int in_channels,
       	int num_packed_channels
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

// feats: M, IC
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// out: M, OC
torch::Tensor gemm_forward_cuda(
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

    if (num_in_channels != _kernel.size(0))
        throw std::invalid_argument("Kernel and input channels mismatch");
    if (num_out_channels != _scaling_factors.size(1))
        throw std::invalid_argument("Invalid scaling factors size (dim1)");
    if (_zeros.size(0) != _scaling_factors.size(0))
        throw std::invalid_argument("Invalid zeros size (dim0)");

    dim3 threads(32, 16);
    dim3 blocks((num_in_feats * num_packed_channels + 512 - 1) / 512, 1);

    _quant_mm<<<blocks, threads>>>(in_feats, kernel, scaling_factors, zeros, out_feats, num_in_channels, num_packed_channels);

    return _out_feats;
}

