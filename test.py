from tqdm import tqdm
import os
import glob

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
        result = new_inf.gemm_forward_cuda(
            ins, qweight, scales, zeros, 8
        )
    else:
        result = awq_inference_engine.gemm_forward_cuda(
            ins, qweight, scales, zeros, 8
        )

    diff = result - outs
    nan_ix = diff.isnan()
    assert ((diff == 0.0) | nan_ix).all().item()


def run_test(folder, new_kernel=False):
    files = glob.glob(os.path.join(folder, '*.pt'))
    for f in tqdm(files):
        test_file(f, new_kernel)
        print("\033[32mPASS: {}\033[0m".format(f))


def dequantize_test():
    print('running dequantize test')
    path = os.path.join(ROOT, 'dequantize.pt')
    data = torch.load(path)

    weights = data['weights']
    scales = data['scales']
    zeros = data['zeros']
    kernel = data['qweight']

    weights = torch.transpose(weights, 0, 1)

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
    assert ((diff == 0.0) | nan_ix).all().item()

def python_dequantize(kernel, scales, zeros):
    """
    proof of concept with python
    """
    return kernel


if __name__ == '__main__':
    dequantize_test()

    folders = glob.glob(ROOT + '/regression_*')
    for new_kernel in [False, True]:
        print('using new kernel: {}'.format(new_kernel))
        for f in folders:
            run_test(f, new_kernel)
        print()
