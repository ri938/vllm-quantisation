
from transformers import LlamaConfig

import os
from safetensors.torch import load_file

from vllm.model_executor.parallel_utils import parallel_state
from vllm.model_executor.layers.gptq import QuantLinear
from functools import partial
import torch
import time


USE_30B = True
# original huggingface weights

if USE_30B:
    WEIGHTS_PATH = '/home/fsuser/dummy/Wizard-Vicuna-30B-Uncensored-fp16'
    filename = 'combined.safetensors'
    folder = '/home/fsuser/dummy/Wizard-Vicuna-30B-Uncensored-GPTQ-Act-Order-False'
    QUANTISED_WEIGHTS = os.path.join(folder, filename)
    NUM_LAYERS = 60
else:
    WEIGHTS_PATH = '/home/fsuser/dummy/Wizard-Vicuna-13B-Uncensored-HF'
    filename = 'Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-g128.safetensors'
    folder = '/home/fsuser/dummy/Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-g128'
    QUANTISED_WEIGHTS = os.path.join(folder, filename)
    NUM_LAYERS = 40


def get_quant_layer(gptq_tensors, N, name):
    assert 0 <= N <= NUM_LAYERS
    qweight = gptq_tensors['model.layers.{}.{}.qweight'.format(N, name)]
    g_idx = gptq_tensors['model.layers.{}.{}.g_idx'.format(N, name)]
    qzeros = gptq_tensors['model.layers.{}.{}.qzeros'.format(N, name)]
    scales = gptq_tensors['model.layers.{}.{}.scales'.format(N, name)]
    return QuantLinear(qweight, qzeros, scales, g_idx, None, 4, 128)


def get_multiple_quant_layer(gptq_tensors, N, names):
    assert 0 <= N <= NUM_LAYERS

    qweights = [gptq_tensors['model.layers.{}.{}.qweight'.format(N, name)] for name in names]
    qzeros = [gptq_tensors['model.layers.{}.{}.qzeros'.format(N, name)] for name in names]
    scales = [gptq_tensors['model.layers.{}.{}.scales'.format(N, name)] for name in names]
    g_idxs = [gptq_tensors['model.layers.{}.{}.g_idx'.format(N, name)] for name in names]

    qweight = torch.cat(qweights, dim=1)
    qzeros = torch.cat(qzeros, dim=1)
    scales = torch.cat(scales, dim=1)

    for item in g_idxs[1:]:
        torch.testing.assert_close(item, g_idxs[0])

    return QuantLinear(qweight, qzeros, scales, g_idxs[0], None, 4, 128)


def calculate_memory_usage(model):
    # doesnt include the peak usage for the forward pass
    # returns in bytes
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    return mem_params + mem_bufs


def quantise_multiple_layers(raw_model, gptq_tensors, names, output_name):
    print('quantising {} to {}...'.format(','.join(names), output_name))
    for pos in range(0, NUM_LAYERS):
        name = 'model.layers.{}.{}'.format(pos, output_name)
        quant_layer = get_multiple_quant_layer(gptq_tensors, pos, names).to(0)
        parent = '.'.join(name.split('.')[1:-1])
        key = output_name.split('.')[-1]
        setattr(raw_model.get_submodule(parent), key, quant_layer)
    return raw_model


def quantise_single_layer(raw_model, gptq_tensors, name):
    print('quantising {}....'.format(name))
    for pos in range(0, NUM_LAYERS):
        target_name = 'model.layers.{}.{}'.format(pos, name)
        quant_layer = get_quant_layer(gptq_tensors, pos, name=name).to(0)
        parent = '.'.join(target_name.split('.')[1:-1])
        key = name.split('.')[-1]
        setattr(raw_model.get_submodule(parent), key, quant_layer)
    return raw_model


def quantise_layers(raw_model):
    print('loading the GPTQ weights')
    print('loading', QUANTISED_WEIGHTS)
    gptq_tensors = load_file(QUANTISED_WEIGHTS)

    gb_in_bytes = 1024 ** 3
    before = calculate_memory_usage(raw_model) / gb_in_bytes
    print('MEMORY BEFORE (GB)', before)

    quantise_single_layer(raw_model, gptq_tensors, 'mlp.down_proj')

    quantise_multiple_layers(
        raw_model,
        gptq_tensors,
        ['mlp.gate_proj', 'mlp.up_proj'],
        'mlp.gate_up_proj'
    )

    quantise_single_layer(raw_model, gptq_tensors, 'self_attn.o_proj')

    quantise_multiple_layers(
        raw_model,
        gptq_tensors,
        ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
        'self_attn.qkv_proj'
    )

    after = calculate_memory_usage(raw_model) / gb_in_bytes
    print('MEMORY AFTER (GB)', after)
    print('% decreases in memory', '{:.4f}'.format(100 * (before - after) / before))
