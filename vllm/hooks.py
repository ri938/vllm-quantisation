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
        match = any([name.endswith(l) for l in layer_names])
        if match:
            fn = partial(hook_fn, name=name)
            layer.register_forward_hook(fn)
    return model
