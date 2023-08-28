from tqdm import tqdm
import os
import glob
import mock
import time

from torch.nn import functional as F
import torch

from safetensors.torch import load_file
from vllm.model_executor.layers import quant
import awq_inference_engine
import new_inf

from enum import Enum

import exllama


ROOT = '/code/test_layer'


class Target(Enum):
    original = 1
    python = 2
    new = 3
    python_new_dequant = 4
    exllama = 5


def load_original_layer(name):
    for filename in glob.glob('/mnt/pvc/Wizard-Vicuna-13B-Uncensored-HF/*.safetensors'):
        data = load_file(filename)
        if name in data:
            return data[name]
    raise ValueError('unable to find weight {}'.format(name))


def find_matching_source_weights(source, ins, zeros):
    name = None
    for key, value in source.items():
        if not key.endswith('zeros'):
            continue
        if value.shape == zeros.shape:
            if (value.to(0) == zeros.to(0)).all():
                name = key
                break
    if name is None:
        # e.g merged laers gate_up
        print('unable to find source weights')
    return name


def test_file(f, target_kernel, source=None):
    """
    We are performing a linear layer on the same
    quantized data so these results should be very close
    if not zero.
    """
    data = torch.load(f)

    ins = data['input']
    outs = data['output']

    scales = data['scales']
    qweight = data['qweight']
    zeros = data['zeros']

    # find the source weights (useful for getting a real comparison)
    if source is not None:
        name = find_matching_source_weights(source, ins, zeros)
        if name is None:
            source = None
        else:
            weights = load_original_layer(name.replace('.qzeros', '.weight'))
            expected = F.linear(ins.to(0), weights.to(0))

    if target_kernel == Target.original:
        result = awq_inference_engine.gemm_forward_cuda(
            ins, qweight, scales, zeros, 8
        )
    elif target_kernel == Target.python:
        result = python_impl(
            ins, qweight, scales, zeros
        )
    elif target_kernel == Target.new:
        result = new_inf.gemm_forward_cuda(
            ins, qweight, scales, zeros, 8
        )
    elif target_kernel == Target.python_new_dequant:
        result = python_impl_cuda_dequant(
            ins, qweight, scales, zeros
        )
    elif target_kernel == Target.exllama:
        result = python_exllam_impl(
            ins, qweight, scales, zeros
        )
    else:
        raise NotImplementedError('unknown kernel {}'.format(target_kernel))

    assert result.shape == outs.shape

    if source is not None:
        assert expected.shape == outs.shape
        diff = result - expected
        nan_ix = diff.isnan()
        mean_diff = diff[~nan_ix].abs().double().mean()
        max_diff = diff[~nan_ix].abs().double().max()
        print('against source mean {} max {}'.format(mean_diff, max_diff))

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
    elif max_diff.item() > 0.0001:
        print('max', max_diff)

    if mean_diff.item() > 0.0005:
        print('mean', mean_diff)
        match = False
    elif mean_diff.item() > 0.00005:
        print('mean', mean_diff)


    #if new_kernel:
    #import pdb; pdb.set_trace()

    return match


def run_test(folder, new_kernel=False, source=None):
    files = glob.glob(os.path.join(folder, '*.pt'))
    for f in tqdm(files):
        match = test_file(f, new_kernel, source=source)
        if match:
            print("\033[32mPASS: {}\033[0m".format(f))
        else:
            print("\033[91mFAIL: {}\033[0m".format(f))
        print()


def python_impl(inputs, kernel, scales, zeros):
    weights = python_dequantize(kernel, scales, zeros)
    res = F.linear(inputs.to(0), weights.to(0))
    return res


def python_impl_cuda_dequant(inputs, kernel, scales, zeros):
    #alt_weights = python_dequantize(kernel, scales, zeros)
    weights = new_inf.dequantize(kernel, scales, zeros)
    #import pdb; pdb.set_trace()
    weights = torch.transpose(weights, 0, 1)
    res = F.linear(inputs.to(0), weights.to(0))
    return res


def dequantize_test():
    """
    We have the processed kernel, scales, zeros and the original weights

    we cannot recover the original weights from the kernels etc.

    The function f(real weights, scales, zeros) -> kernel is not known
     and convoluted so cant recover and check this
    """
    print('running dequantize test')
    path = os.path.join(ROOT, 'dequantize.pt')
    data = torch.load(path)

    weights = data['weights']
    scales = data['scales']
    zeros = data['zeros']
    kernel = data['qweight']

    print('checking packing algorithm on kernel')
    repacked = pack(unpack(kernel))
    assert (repacked == kernel).all()

    print('checking packing algorithm on zeros')
    repacked = pack(unpack(zeros))
    assert (repacked == zeros).all()

    """
    dq = new_inf.dequantize(
        kernel,
        scales,
        zeros
    )
    """

    # first we check the dequantized weights against real weights
    # no reason they should be the same
    dq = python_dequantize(kernel.to(0), scales.to(0), zeros.to(0))
    assert dq.shape == weights.shape

    diff = dq - weights.to(0)
    nan_ix = diff.isnan()

    print('dequantized diff with original weights (shouldnt match)')
    print('mean', diff.abs().double().mean())

    return dq


def python_exllam_impl(ins, qweight, scales, zeros):
    """Its just not the same shape: qweights is reshaped """
    gptq = repack_gptq(qweight)

    none_tensor = torch.empty((1, 1), device = "meta")

    import pdb; pdb.set_trace()
    q4 = exllama.make_q4(gptq, zeros, scales, none_tensor, 0)

    q4_width = qweight.shape[1] * 8  # ?
    output = torch.empty((ins.shape[0], q4_width), dtype=torch.float16, device=ins.device)
    exllama.q4_matmul(ins.to(0), q4, output)
    return output


def unpack(matrix, dtype=torch.int32):
    output = torch.zeros((matrix.shape[0], matrix.shape[1] * 8), dtype=dtype)

    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    for column in range(matrix.shape[1]):
        for idx, position in enumerate(order_map):
            output[:, column * 8 + position] = matrix[:, column] >> (4 * idx) & 0b1111

    return output


def pack(matrix, dtype=torch.int32):
    output = torch.zeros((matrix.shape[0], matrix.shape[1] // 8), dtype=dtype)

    pack_num = 32 // 4

    for col in range(matrix.shape[1] // pack_num):
        order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        for i in range(pack_num):
            matrix_col = matrix[:, col * pack_num + order_map[i]]
            output[:, col] |= matrix_col << (i * 4)

    return output


def repack_gptq(matrix, dtype=torch.int32):
    unpacked = unpack(matrix).to(0)
    repacked = pack_gptq(unpacked)
    return repacked


def pack_gptq(matrix, dtype=torch.int32):
    output = torch.zeros((matrix.shape[0], matrix.shape[1]), dtype=dtype).to(0)

    pack_num = 32 // 4

    for col in range(matrix.shape[1] // pack_num):
        for i in range(pack_num):
            matrix_col = unpacked[:, col * pack_num + i]
            output[:, col] |= matrix_col << (i * 4)

    return output


def python_dequantize(kernel, scales, zeros):
    """
    I dont think this works: its a more different function
    to recover the original weights
    """
    in_feats, in_channels = kernel.shape[0], kernel.shape[1] * 8

    new_zeros = unpack(zeros, dtype=torch.int32)
    assert new_zeros.shape[1] == in_channels
    assert new_zeros.shape[0] * 128 == in_feats

    assert scales.shape == new_zeros.shape

    scale_zeros = new_zeros.to(0) * scales

    new_kernel = unpack(kernel, dtype=torch.int32)
    assert new_kernel.shape == torch.Size([in_feats, in_channels])
    new_kernel = torch.transpose(new_kernel, 0, 1).to(0)

    output = []

    for column in range(in_feats):
        output.append((new_kernel[:, column] * scales[column // 128]) - scale_zeros[column // 128])

    res = torch.stack(output, dim=1)
    return res.to(torch.float16)


if __name__ == '__main__':
    #dequantize_test()
    test_cases = [
        Target.python_new_dequant,
        Target.original,
        Target.python,
        #Target.new,
        #Target.exllama,
    ]

    source = load_file('/code/wizard-vicuna-13b-uncensored-awq-4bit-g128/wizard-vicuna-13b-w4-g128-awq.safetensors')

    folders = glob.glob(ROOT + '/regression_*')
    for new_kernel in test_cases:
        print('using new kernel: {}'.format(new_kernel))
        start = time.time()
        for f in folders:
            run_test(f, new_kernel, source)
        print('duration', time.time() - start)
        print()
