
from transformers import LlamaConfig

import os
from safetensors.torch import load_file
import json

from vllm.model_executor.models import llama
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils import parallel_state
from vllm.model_executor.layers.gptq import QuantLinear
from vllm.model_executor.layers.exllama import Ex4bitLinear, SuperLayer
from functools import partial
import torch
import time

from vllm.model_executor.layers.exllama import create_exllama_buffers, set_device
from torch.profiler import profile, record_function, ProfilerActivity


# original huggingface weights
WEIGHTS_PATH = '/mnt/models/Wizard-Vicuna-13B-Uncensored-HF'

# quantised weights
filename = 'Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-g128.safetensors'
folder = '/mnt/pvc/Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-g128'
QUANTISED_WEIGHTS = os.path.join(folder, filename)

PROFILE = False
PROFILE_STACKTRACE = False

EXLLAMA = True


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


def get_linear(qweight, qzeros, scales, g_idx):
    if EXLLAMA:
        qweight = qweight.to(0)
        qzeros = qzeros.to(0)
        scales = scales.to(0)
        g_idx = g_idx.to(0)
        #layer = Ex4bitLinear(qweight, qzeros, scales, g_idx, None, 4, 128)
        layer = SuperLayer(qweight, qzeros, scales, g_idx)
    else:
        layer = QuantLinear(qweight, qzeros, scales, g_idx, None, 4, 128).to(0)

    return layer


def initialise_exllama():
    # initialising exllama buffers
    print('initialising exllama')
    set_device(torch.device('cuda:0'))
    create_exllama_buffers()


def get_quant_layer(gptq_tensors, N, name):
    assert 0 <= N <= 39
    qweight = gptq_tensors['model.layers.{}.{}.qweight'.format(N, name)]
    g_idx = gptq_tensors['model.layers.{}.{}.g_idx'.format(N, name)]
    qzeros = gptq_tensors['model.layers.{}.{}.qzeros'.format(N, name)]
    scales = gptq_tensors['model.layers.{}.{}.scales'.format(N, name)]
    return get_linear(qweight, qzeros, scales, g_idx)


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

    return get_linear(qweight, qzeros, scales, g_idxs[0])


def compare_raw_modules(target_layers, quant_layers, examples):
    print('how did we do?')
    for layer in range(40):
        print('#### layer', layer, '####')
        exs = examples[layer][0, :]

        target = forward_example(target_layers[layer], exs)
        found = get_quant_layer(gptq_tensors, layer).forward(exs)
        
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


def print_example_responses(model, realistic=False):
    if realistic:
        example_inputs = [s['model_input'][-2048:] for s in read_jsonlines()]
    else:
        example_inputs = [
            'AI is going to',
            'The capitol of France is',
            'Now this is a story all about how, my',
            'This is a story about a frog who',
        ]

    for line in example_inputs:
        if PROFILE:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    test_response(model, line)
            averages = prof.key_averages(group_by_input_shape=True)
            print(averages.table(sort_by="cuda_time_total", row_limit=50))
        elif PROFILE_STACKTRACE:
            # to allow viewing with tensorboard
            print('saving detailed profiler')
            prof = torch.profiler.profile(
                #schedule=torch.profiler.schedule(),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler'),
                record_shapes=True,
                with_stack=True)
            prof.start()
            test_response(model, line)
            prof.stop()
        else:
            test_response(model, line)
        print()


@torch.inference_mode()
def test_response(model, line):
    print('input:', line)
    start = time.time()
    params = SamplingParams(n=1, max_tokens=128, logprobs=10)
    resp = model.generate(line, params, use_tqdm=False)[0].outputs
    duration = time.time() - start
    num_new_tokens = [len(r.token_ids) for r in resp]
    print('output:', [r.text for r in resp])
    print('token / second', max(num_new_tokens) / duration)
    print('duration', duration)


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

    after = calculate_memory_usage(model.llm_engine.workers[0].model) / gb_in_bytes
    print('MEMORY AFTER (GB)', after)
    print('% decreases in memory', '{:.4f}'.format(100 * (before - after) / before))


def read_jsonlines():
    path = 'example_requests.json'
    lines = []
    with open(path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


if __name__ == '__main__':
    model = get_basic_model()
    raw_model = model.llm_engine.workers[0].model.model

    quantise_layers(raw_model)
    print(raw_model)

    if EXLLAMA:
        initialise_exllama()

    print('#### AFTER ####')
    print_example_responses(model, realistic=True)
