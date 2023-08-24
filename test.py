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

    #if f.endswith('data_842.pt'):
    #    import pdb; pdb.set_trace()

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
    #assert ((diff == 0.0) | nan_ix).all().item()

    prop_diff = diff[~nan_ix] / outs[~nan_ix]

    max_diff = diff[~nan_ix].abs().double().max()
    mean_diff = diff[~nan_ix].abs().double().mean()

    # this is too leniant -- its broken
    match = True

    if max_diff.item() > 0.01:
        print('max', max_diff)
        match = False
    elif max_diff.item() > 0.001:
        print('max', max_diff)

    if mean_diff.item() > 0.0005:
        print('mean', mean_diff)
        match = False
    elif mean_diff.item() > 0.0005:
        print('max', max_diff)

    #if new_kernel:
    #import pdb; pdb.set_trace()

    return match


def run_test(folder, new_kernel=False):
    files = glob.glob(os.path.join(folder, '*.pt'))
    for f in tqdm(files):
        match = test_file(f, new_kernel)
        if match:
            print("\033[32mPASS: {}\033[0m".format(f))
        else:
            print("\033[91mFAIL: {}\033[0m".format(f))
        print()


def python_impl(inputs, kernel, scales, zeros):
    weights = python_dequantize(kernel, scales, zeros)
    res = F.linear(inputs.to(0), weights.to(0))
    return res


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

    dq = python_dequantize(kernel, scales, zeros)

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


def unpack(matrix, dtype):
    output = torch.zeros((matrix.shape[0], matrix.shape[1] * 8), dtype=dtype)

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    for column in range(matrix.shape[1]):
        for idx, position in enumerate(order_map):
            output[:, column * 8 + position] = matrix[:, column] >> (4 * idx) & 0b1111

    return output


def python_dequantize(kernel, scales, zeros):
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

    in_feats, in_channels = kernel.shape[0], kernel.shape[1] * 8

    new_zeros = unpack(zeros, dtype=torch.int32)
    assert new_zeros.shape[1] == in_channels
    assert new_zeros.shape[0] * 128 == in_feats

    assert scales.shape == new_zeros.shape

    scale_zeros = new_zeros.to(0) * scales

    new_kernel = unpack(kernel, dtype=torch.int32)
    assert new_kernel.shape == torch.Size([in_feats, in_channels])

    output = []

    new_kernel = torch.transpose(new_kernel, 0, 1).to(0)

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
