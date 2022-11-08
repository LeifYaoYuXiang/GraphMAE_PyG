import numpy as np
from model_v1 import PreTrainModel
import torch
from torch import optim as optim
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


def build_model(model_config, train_test_config):
    # build model
    model = PreTrainModel(model_config)
    # build optimizer
    optimizer_name = model_config['optimizer_name']
    optimizer_name = optimizer_name.lower()
    parameters = model.parameters()
    optimizer_args = dict(lr=train_test_config['lr'], weight_decay=train_test_config['weight_decay'])
    if optimizer_name == "adam":
        optimizer = optim.Adam(parameters, **optimizer_args)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(parameters, **optimizer_args)
    elif optimizer_name == "adadelta":
        optimizer = optim.Adadelta(parameters, **optimizer_args)
    elif optimizer_name == "radam":
        optimizer = optim.RAdam(parameters, **optimizer_args)
    elif optimizer_name == "sgd":
        optimizer_args["momentum"] = 0.9
        optimizer = optim.SGD(parameters, **optimizer_args)
    else:
        return NotImplementedError
    # build scheduler
    scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / train_test_config['max_epoch'])) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

    if model_config['pooling'] == 'max':
        pooler = global_max_pool
    elif model_config['pooling'] == 'mean':
        pooler = global_mean_pool
    elif model_config['pooling'] == 'sum':
        pooler = global_add_pool
    else:
        raise NotImplementedError

    return model, optimizer, scheduler, pooler
