
from transformers import LlamaConfig

import os
from safetensors.torch import load_file

from vllm.model_executor.models import llama

from vllm.model_executor.parallel_utils import parallel_state
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
