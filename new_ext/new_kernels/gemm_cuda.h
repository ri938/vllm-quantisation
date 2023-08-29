#include <torch/extension.h>

torch::Tensor gemm_forward_cuda_new(torch::Tensor _in_feats, torch::Tensor _kernel,
    torch::Tensor _scaling_factors, torch::Tensor _zeros);

torch::Tensor dequantize(torch::Tensor _kernel,
    torch::Tensor _scaling_factors, torch::Tensor _zeros);
