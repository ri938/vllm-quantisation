
from transformers import LlamaConfig

import os
from safetensors.torch import load_file

from vllm.model_executor.parallel_utils import parallel_state
from vllm.model_executor.layers.gptq import QuantLinear
from vllm.awq.quantize import qmodule

from functools import partial
import torch
import time


QUANTISED_WEIGHTS = '/code/quant_cache/wizard-vicuna-13b-w4-g128-awq.safetensors'


def get_quant_layer(gptq_tensors, N, name):
    assert 0 <= N <= 39
    qweight = gptq_tensors['model.layers.{}.{}.qweight'.format(N, name)]
    qzeros = gptq_tensors['model.layers.{}.{}.qzeros'.format(N, name)]
    scales = gptq_tensors['model.layers.{}.{}.scales'.format(N, name)]

    in_features = qweight.shape[0]
    out_features = qweight.shape[1] * 8   ## 32 // w_bit

    return get_linear(qweight, qzeros, scales, in_features, out_features)


def get_multiple_quant_layer(gptq_tensors, N, names):
    assert 0 <= N <= 39

    qweights = [gptq_tensors['model.layers.{}.{}.qweight'.format(N, name)] for name in names]
    qzeros = [gptq_tensors['model.layers.{}.{}.qzeros'.format(N, name)] for name in names]
    scales = [gptq_tensors['model.layers.{}.{}.scales'.format(N, name)] for name in names]

    qweight = torch.cat(qweights, dim=1)
    qzeros = torch.cat(qzeros, dim=1)
    scales = torch.cat(scales, dim=1)

    return get_linear(qweight, qzeros, scales)


def calculate_memory_usage(model):
    # doesnt include the peak usage for the forward pass
    # returns in bytes
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    return mem_params + mem_bufs


def quantise_multiple_layers(raw_model, gptq_tensors, names, output_name):
    print('quantising {} to {}...'.format(','.join(names), output_name))
    for pos in range(0, 40):
        name = 'model.layers.{}.{}'.format(pos, output_name)
        quant_layer = get_multiple_quant_layer(gptq_tensors, pos, names)
        parent = '.'.join(name.split('.')[1:-1])
        key = output_name.split('.')[-1]
        setattr(raw_model.get_submodule(parent), key, quant_layer)
    return raw_model


def quantise_single_layer(raw_model, gptq_tensors, name):
    print('quantising {}....'.format(name))
    for pos in range(0, 40):
        target_name = 'model.layers.{}.{}'.format(pos, name)
        quant_layer = get_quant_layer(gptq_tensors, pos, name=name)
        parent = '.'.join(target_name.split('.')[1:-1])
        key = name.split('.')[-1]
        setattr(raw_model.get_submodule(parent), key, quant_layer)
    return raw_model


def load_weights(raw_model, gptq_tensors, name):
    print('load weights {}....'.format(name))
    for pos in range(0, 40):
        target_name = 'model.layers.{}.{}'.format(pos, name)
        weights = gptq_tensors[target_name + '.weight']
        path = '.'.join(target_name.split('.')[1:])
        setattr(raw_model.get_submodule(path), 'weight', torch.nn.Parameter(weights))
    return raw_model


def quantise_layers(raw_model):
    print('loading the GPTQ weights')
    print('loading', QUANTISED_WEIGHTS)
    gptq_tensors = load_file(QUANTISED_WEIGHTS)

    gb_in_bytes = 1024 ** 3
    before = calculate_memory_usage(raw_model) / gb_in_bytes
    print('MEMORY BEFORE (GB)', before)

    quantise_single_layer(raw_model, gptq_tensors, 'mlp.down_proj')
    quantise_single_layer(raw_model, gptq_tensors, 'mlp.up_proj')
    quantise_single_layer(raw_model, gptq_tensors, 'mlp.gate_proj')

    load_weights(raw_model, gptq_tensors, 'input_layernorm')
    load_weights(raw_model, gptq_tensors, 'post_attention_layernorm')

    #quantise_single_layer(raw_model, gptq_tensors, 'self_attn.o_proj')
    #quantise_single_layer(raw_model, gptq_tensors, 'self_attn.k_proj')
    #quantise_single_layer(raw_model, gptq_tensors, 'self_attn.v_proj')
    #quantise_single_layer(raw_model, gptq_tensors, 'self_attn.q_proj')

    after = calculate_memory_usage(raw_model) / gb_in_bytes
    print('MEMORY AFTER (GB)', after)
    print('% decreases in memory', '{:.4f}'.format(100 * (before - after) / before))


def get_linear(qweight, qzeros, scales, in_features, out_features):
    layer = qmodule.WQLinear(
        w_bit=4,
        group_size=128,
        in_features=in_features,
        out_features=out_features,
        bias=None,
        dev=0
    )

    assert layer.qweight.shape == qweight.shape
    assert layer.qweight.dtype == qweight.dtype
    layer.qweight = qweight

    assert layer.qzeros.shape == qzeros.shape
    assert layer.qzeros.dtype == qzeros.dtype
    layer.qzeros = qzeros

    assert layer.scales.shape == scales.shape
    assert layer.scales.dtype == scales.dtype
    layer.scales = scales

    #import pdb; pdb.set_trace()

    return layer
