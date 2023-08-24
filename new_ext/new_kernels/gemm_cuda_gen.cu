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


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// in_feats: M, IC [float16]
// kernel: IC, OC // 8 [int32] -> cast to IC, OC [uint4b]
// scaling_factors: IC // G, OC [float16]
// zeros: IC // G, OC // 8 [int32] -> cast to IC // G, OC [uint4b]
// assume that batch_size < 16 for now


torch::Tensor dequantize(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros)
{
    int num_in_feats = _kernel.size(0);
    int num_in_channels = _kernel.size(1) * 8;
    int num_packed_columns = _kernel.size(1);

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(_kernel.device());
    at::Tensor dq = torch::zeros({num_in_feats, num_in_channels}, options);

    for (int row = 0; row < num_in_feats; row++) {
         for (int column = 0; column < num_packed_columns; column++) {

         }
    }

    return dq;
}

   
torch::Tensor gemm_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_in_feats));

    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::zeros({split_k_iters, num_in_feats, _kernel.size(1) * 8}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);

    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());
    int group_size = num_in_channels / _scaling_factors.size(0);

    if (num_out_channels % 64 != 0)
        throw std::invalid_argument("OC is not multiple of cta_N = 64");
    if (num_out_channels % 8 != 0)
        throw std::invalid_argument("OC is not multiple of pack_num = 8");
    if (group_size % 32 != 0)
	throw std::invalid_argument("Group size should be a multiple of 32");
    if (num_out_channels % group_size != 0)
        throw std::invalid_argument("OC is not multiple of Group size");

    // level 1: we just want to dequantize the kernel

    torch::Tensor dq = dequantize(
	_kernel,
	_scaling_factors,
	_zeros
    );

    // level 2: use standard libraries in order to matrix multiply it
    // level 3: optimise

    return _out_feats.sum(0);
}
