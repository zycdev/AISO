from typing import List

import torch
from torch import nn


def get_device(model):
    return next(model.parameters()).device


def init_weights(modules: List):
    for module in modules:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def load_state(model, path, exact=True, strict=True):
    state_dict = torch.load(path, map_location=torch.device('cpu'))

    def filter_name(x):
        return x[7:] if x.startswith('module.') else x

    if exact:
        state_dict = {filter_name(k): v for (k, v) in state_dict.items()}
    else:
        state_dict = {filter_name(k): v for (k, v) in state_dict.items() if filter_name(k) in model.state_dict()}
    model.load_state_dict(state_dict, strict=strict)
    return model


def save_model(model, path):
    model_to_save = model.module if hasattr(model, 'module') else model
    if torch.__version__ >= '1.4':
        torch.save(model_to_save.state_dict(), path, _use_new_zipfile_serialization=False)
    else:
        torch.save(model_to_save.state_dict(), path)
