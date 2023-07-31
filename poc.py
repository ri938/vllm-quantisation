
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


WEIGHTS_PATH = '/mnt/models/Wizard-Vicuna-13B-Uncensored-HF'


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
    #config_path = '/code/Wizard-Vicuna-7B-Uncensored-HF/config.json'

    config = LlamaConfig.from_json_file(config_path)

    torch.set_default_dtype(torch.float16)
    model = llama.LlamaForCausalLM(config)

    print('loading weights from the original model')
    model.load_weights(WEIGHTS_PATH)
    return model


def get_basic_model():
    print('loading basic model')
    return LLM(WEIGHTS_PATH)


model = get_basic_model()


print('loading the GPTQ weights')
filename = 'Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-g128.safetensors'
folder = '/mnt/pvc/Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-g128'
path = os.path.join(folder, filename)
print('loading', path)

# dict from name to value
gptq_tensors = load_file(path)

# RowParallelLinear
raw_model = model.llm_engine.workers[0].model.model
target_layers = [l.mlp.down_proj for l in raw_model.layers]


examples = torch.rand((10, 13824), dtype=torch.float16).to(0)
#examples = torch.rand((10, 11008), dtype=torch.float16).to(0)

examples = {i: torch.load('/tmp/layers.{}.mlp.down_proj.torch'.format(i))[0].to(0) for i in range(0, 40)}


@torch.inference_mode()
def forward_example(layer, examples):
     return layer.forward(examples)[0]


def get_quant_layer(gptq_tensors, N):
    assert 0 <= N <= 39
    qweight = gptq_tensors['model.layers.{}.mlp.down_proj.qweight'.format(N)]
    g_idx = gptq_tensors['model.layers.{}.mlp.down_proj.g_idx'.format(N)]
    qzeros = gptq_tensors['model.layers.{}.mlp.down_proj.qzeros'.format(N)]
    scales = gptq_tensors['model.layers.{}.mlp.down_proj.scales'.format(N)]
    return QuantLinear(qweight, qzeros, scales, g_idx, None, 4, 128)


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


def merge_quantised_layers(raw_model, gptq_tensors):
    for pos in range(0, 40):
        name = 'model.layers.{}.mlp.down_proj'.format(pos)
        #print('replacing layer', name)
        quant_layer = get_quant_layer(gptq_tensors, pos).to(0)
        parent = '.'.join(name.split('.')[1:-1])
        raw_model.get_submodule(parent).down_proj = quant_layer
    return raw_model


@torch.inference_mode()
def print_example_responses(model):
    example_inputs = [
        'AI is going to',
        'The capitol of France is',
        'Now this is a story all about how, my',
        'This is a story about a frog who',
    ]

    for line in example_inputs:
        print('input:', line)
        start = time.time()
        params = SamplingParams(max_tokens=128, logprobs=10)
        resp = model.generate(line, params, use_tqdm=False)[0].outputs[0]
        duration = time.time() - start
        num_new_tokens = len(resp.token_ids)
        print('output:', resp.text)
        print('token / second', num_new_tokens / duration)
        print('duration', duration)
        print()

print('#### BEFORE ####')
#print_example_responses(model)

print()

gb_in_bytes = 1024 ** 3
print('MEMORY BEFORE', calculate_memory_usage(model.llm_engine.workers[0].model) / gb_in_bytes)
merge_quantised_layers(raw_model, gptq_tensors)
print('MEMORY AFTER', calculate_memory_usage(model.llm_engine.workers[0].model) / gb_in_bytes)

print()

print('#### AFTER ####')
print_example_responses(model)
