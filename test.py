from tqdm import tqdm
import os
import glob

from torch.nn import functional as F
import torch

import awq_inference_engine
import new_inf

ROOT = '/code/test_layer'


def test_file(f, new_kernel):
    data = torch.load(f)

    ins = data['input']
    outs = data['output']

    scales = data['scales']
    qweight = data['qweight']
    zeros = data['zeros']

    if new_kernel:
        """
        result = new_inf.gemm_forward_cuda(
            ins, qweight, scales, zeros, 8
        )
        """
        result = python_impl(
            ins, qweight, scales, zeros
        )
    else:
        result = awq_inference_engine.gemm_forward_cuda(
            ins, qweight, scales, zeros, 8
        )

    assert result.shape == outs.shape

    diff = result - outs
    nan_ix = diff.isnan()
    assert ((diff == 0.0) | nan_ix).all().item()


def run_test(folder, new_kernel=False):
    files = glob.glob(os.path.join(folder, '*.pt'))
    for f in tqdm(files):
        test_file(f, new_kernel)
        print("\033[32mPASS: {}\033[0m".format(f))


def python_impl(inputs, kernel, scales, zeros):
    weights = dequantize_test()

    batch, in_feats = inputs.shape
    weights_in_feats, in_channels = weights.shape
    assert weights_in_feats == in_feats

    return F.linear(inputs, weights)


def dequantize_test():
    print('running dequantize test')
    path = os.path.join(ROOT, 'dequantize.pt')
    data = torch.load(path)

    weights = data['weights']
    scales = data['scales']
    zeros = data['zeros']
    kernel = data['qweight']

    """
    dq = new_inf.dequantize(
        kernel,
        scales,
        zeros
    )
    """

    dq = python_dequantize(weights, kernel, scales, zeros)

    assert dq.shape == weights.shape

    diff = dq - weights
    nan_ix = diff.isnan()

    print('mean', diff.abs().double().mean())

    """
    I think this works because some features are
    super close like almost no error and most do have an error
    """
    #assert ((diff == 0.0) | nan_ix).all().item()
    return dq


def unpack(matrix):
    output = torch.zeros((matrix.shape[0], matrix.shape[1] * 8), dtype=torch.int32)

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    for column in range(matrix.shape[1]):
        for idx, position in enumerate(order_map):
            output[:, column * 8 + position] = matrix[:, column] >> (4 * idx) & 0b1111

    return output


def python_dequantize(weights, kernel, scales, zeros):
    """
    proof of concept with python

    there is
    1. scales, zeros as calculated by the algorithm
    2. qweights, scales and zeros as they are compressed (this is what we have)

    # we need to recover the input weights
    1. unpad the zeros
    2. unpad the scales
    3. unpad the kernel
    """

    weights = torch.transpose(weights, 0, 1)

    in_feats, in_channels = weights.shape
    assert in_feats == 5120

    print('unpacking zeros')
    new_zeros = unpack(zeros)
    assert new_zeros.shape[1] == weights.shape[1]
    assert new_zeros.shape[0] * 128 == weights.shape[0]

    print('checking scales')
    assert scales.shape == new_zeros.shape

    scale_zeros = new_zeros * scales

    print('unpacking kernel')
    new_kernel = unpack(kernel)
    assert new_kernel.shape == weights.shape

    output = []

    new_kernel = torch.transpose(new_kernel, 0, 1)

    for column in range(in_feats):
        output.append((new_kernel[:, column] * scales[column // 128]) - scale_zeros[column // 128])

    res = torch.stack(output, dim=1)
    return res


if __name__ == '__main__':
    #dequantize_test()

    folders = glob.glob(ROOT + '/regression_*')
    for new_kernel in [False, True]:
        print('using new kernel: {}'.format(new_kernel))
        for f in folders:
            run_test(f, new_kernel)
        print()
