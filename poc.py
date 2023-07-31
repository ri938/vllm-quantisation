
from transformers import LlamaConfig

import os
from safetensors.torch import load_file

from vllm.model_executor.models import llama
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils import parallel_state
from vllm.model_executor.layers.gptq import QuantLinear
from functools import partial
import torch
import time


# original huggingface weights
WEIGHTS_PATH = '/mnt/models/Wizard-Vicuna-13B-Uncensored-HF'

# quantised weights
filename = 'Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-g128.safetensors'
folder = '/mnt/pvc/Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-g128'
QUANTISED_WEIGHTS = os.path.join(folder, filename)


def get_raw_model():
    print('init distributed mode')
    port = 8011
    distributed_init_method = f"tcp://localhost:{port}"

    torch.distributed.init_process_group(
        'nccl',
        world_size=1,
        rank=0,
        init_method=distributed_init_method
    )

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

    config_path = '/mnt/models/Wizard-Vicuna-13B-Uncensored-HF/config.json'
    config_path = os.path.join(WEIGHTS_PATH, config_path)

    config = LlamaConfig.from_json_file(config_path)

    torch.set_default_dtype(torch.float16)
    model = llama.LlamaForCausalLM(config)

    print('loading weights from the original model')
    model.load_weights(WEIGHTS_PATH)
    return model


def get_basic_model():
    print('loading basic model')
    return LLM(WEIGHTS_PATH)



@torch.inference_mode()
def forward_example(layer, examples):
     return layer.forward(examples)[0]


def get_quant_layer(gptq_tensors, N, name):
    assert 0 <= N <= 39
    qweight = gptq_tensors['model.layers.{}.{}.qweight'.format(N, name)]
    g_idx = gptq_tensors['model.layers.{}.{}.g_idx'.format(N, name)]
    qzeros = gptq_tensors['model.layers.{}.{}.qzeros'.format(N, name)]
    scales = gptq_tensors['model.layers.{}.{}.scales'.format(N, name)]
    return QuantLinear(qweight, qzeros, scales, g_idx, None, 4, 128)


def get_multiple_quant_layer(gptq_tensors, N, names):
    assert 0 <= N <= 39

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


def compare_raw_modules(target_layers, quant_layers, examples):
    print('how did we do?')
    for layer in range(40):
        print('#### layer', layer, '####')
        exs = examples[layer][0, :]

        target = forward_example(target_layers[layer], exs)
        found = get_quant_layer(gptq_tensors, layer).to(0).forward(exs)
        
        # added an extra None to make compatible. what is going on here?
        if len(found) > 1:
            found = found[0]

        diff = target - found

        print('mean diff', diff.abs().mean().cpu().item())
        print('max diff', diff.abs().max().cpu().item())
        print('raw', diff.abs().cpu())
        print('target', target.cpu())
        print('found', found.cpu())
        print()


def save_tensors(module, inputs, output, name):
    path = '/tmp/{}.torch'.format(name)
    print('saving tensors to path', path)
    torch.save(inputs, path)


def calculate_memory_usage(model):
    # doesnt include the peak usage for the forward pass
    # returns in bytes
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    return mem_params + mem_bufs


def add_save_tensor_hook(raw_model):
    print('all layers')
    for name, layer in raw_model.named_modules():
        #print(name, layer)
        if name.endswith('down_proj'):
            print('registering hook for {}'.format(name))
            fn = partial(save_tensors, name=name)
            layer.register_forward_hook(fn)


def quantise_multiple_layers(raw_model, gptq_tensors, names, output_name):
    print('quantising {} to {}...'.format(','.join(names), output_name))
    for pos in range(0, 40):
        name = 'model.layers.{}.{}'.format(pos, output_name)
        quant_layer = get_multiple_quant_layer(gptq_tensors, pos, names).to(0)
        parent = '.'.join(name.split('.')[1:-1])
        key = output_name.split('.')[-1]
        setattr(raw_model.get_submodule(parent), key, quant_layer)
    return raw_model


def quantise_single_layer(raw_model, gptq_tensors, name):
    print('quantising {}....'.format(name))
    for pos in range(0, 40):
        target_name = 'model.layers.{}.{}'.format(pos, name)
        quant_layer = get_quant_layer(gptq_tensors, pos, name=name).to(0)
        parent = '.'.join(target_name.split('.')[1:-1])
        key = name.split('.')[-1]
        setattr(raw_model.get_submodule(parent), key, quant_layer)
    return raw_model


def print_example_responses(model):
    example_inputs = [
        'AI is going to',
        'The capitol of France is',
        'Now this is a story all about how, my',
        'This is a story about a frog who',
    ]

    for line in example_inputs:
        test_response(model, line)
        print()


@torch.inference_mode()
def test_response(model, line):
    print('input:', line)
    start = time.time()
    params = SamplingParams(max_tokens=128, logprobs=10)
    resp = model.generate(line, params, use_tqdm=False)[0].outputs[0]
    duration = time.time() - start
    num_new_tokens = len(resp.token_ids)
    print('output:', resp.text)
    print('token / second', num_new_tokens / duration)
    print('duration', duration)


if __name__ == '__main__':
    model = get_basic_model()
    raw_model = model.llm_engine.workers[0].model.model

    #target_layers = [l.mlp.down_proj for l in raw_model.layers]
    #examples = {i: torch.load('/tmp/layers.{}.mlp.down_proj.torch'.format(i))[0].to(0) for i in range(0, 40)}

    print('loading the GPTQ weights')
    print('loading', QUANTISED_WEIGHTS)
    gptq_tensors = load_file(QUANTISED_WEIGHTS)

    print('#### BEFORE ####')
    #print_example_responses(model)

    print()

    gb_in_bytes = 1024 ** 3
    before = calculate_memory_usage(model.llm_engine.workers[0].model) / gb_in_bytes
    print('MEMORY BEFORE (GB)', before)

    print('#### quantising layers ####')
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

    print(raw_model)

    after = calculate_memory_usage(model.llm_engine.workers[0].model) / gb_in_bytes
    print('MEMORY AFTER (GB)', after)
    print('% decreases in memory', '{:.4f}'.format(100 * (before - after) / before))

    print()

    print('#### AFTER ####')
    print_example_responses(model)
