from functools import partial
import os

import torch


def pdb_hook(module, input, output, name):
    import pdb; pdb.set_trace()


def print_hook(module, input, output, name):
    data = {
        'name': name,
        'input': {'sum': input[0].sum().item()},
        'output': {'sum': output.sum().item()}
    }
    print(data)


def save_file_hook(module, input, output, name, folder):
    path = os.path.join(folder, name)

    data = {
        'input': input,
        'output': output
    }

    torch.save(data, path)


def add_hooks_to_layers(hook_fn, layer_names, model):
    for name, layer in model.named_modules():
        match = any([name.endswith(l) or l == '*' for l in layer_names])
        if match:
            fn = partial(hook_fn, name=name)
            layer.register_forward_hook(fn)
    return model


def combined_tensors(raw_model_path, awq_path):
    """Create single store of weights. """
    """Dont need this because its already merged. """
    files = glob.glob(os.path.join(raw_model_path, '*.bin'))

    data = {}
    for f in files:
        print(f)
        file_data = torch.load(f)

        skip_layers = [
            'q_proj', 'k_proj', 'v_proj', 'gate_proj',
            'up_proj', 'down_proj', 'o_proj',
            'input_layernorm', 'post_attention_layernorm'
        ]

        for key, value in file_data.items():
            if any([l in key for l in skip_layers]):
                continue
            data[key] = value

