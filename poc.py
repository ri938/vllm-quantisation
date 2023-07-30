
from transformers import LlamaConfig

import os
from safetensors.torch import load_file

from vllm.model_executor.models import llama

from vllm.model_executor.parallel_utils import parallel_state
from vllm.model_executor.layers.gptq import QuantLinear
import torch

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

config = LlamaConfig.from_json_file(config_path)

torch.set_default_dtype(torch.float16)
model = llama.LlamaForCausalLM(config)

weights_path = '/mnt/models/Wizard-Vicuna-13B-Uncensored-HF'

print('loading weights from the original model')
model.load_weights(weights_path)


print('loading the GPTQ weights')
filename = 'Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-g128.safetensors'
folder = '/mnt/pvc/Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-g128'
path = os.path.join(folder, filename)
print('loading', path)

# dict from name to value
gptq_tensors = load_file(path)

# RowParallelLinear
target_layers = [l.mlp.down_proj for l in model.model.layers]


examples = torch.rand((10, 13824), dtype=torch.float16).to(0)


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
    target = forward_example(target_layers[layer], examples)
    found = forward_example(target_layers[layer], examples)

    diff = target - found
    print(layer)
    print('mean diff', diff.mean().cpu().item())
    print('max diff', diff.max().cpu().item())
    print()

# g_idx, qweight, qzeros, scales   model.layers.N.down_proj.X for N in range(0, 40)
