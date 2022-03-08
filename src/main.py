import torch
import config
import random
import logging
import numpy as np
from functools import partial
from models import simple_model
from typing import NamedTuple, Dict
from dataset_iterators import generate_data_iterators
from utils.misc import resolve_device, set_seed, make_opt, CustomError

# Setting up logger
logger = logging.getLogger(__name__)

class RunnerArguments(NamedTuple):
    """Arguments for the main function"""
    seed: int
    dataset_name: str
    batch_size: int
    model: str
    epochs: int = 20
    adv_loss_scale: float = 0.0
    save_model_as: str = 'dummy'
    method: str = 'unconstrained'
    optimizer_name: str = 'adam'
    lr: float = 0.001
    use_lr_schedule: bool = False


def get_model(model_name:str, other_meta_data:Dict, device:torch.device):
    if model_name == 'simple_non_linear':
        model_arch = {
            'encoder': {
                'input_dim': other_meta_data['input_dim'],
                'output_dim': other_meta_data['number_of_main_task_label']
            }
        }

        model_params = {
            'model_arch': model_arch,
            'device': device
        }
        model = simple_model.SimpleNonLinear(model_params)
    else:
        raise NotImplementedError

    return model


def get_optimizer(optimizer_name:str):
    if optimizer_name.lower() == 'adagrad':
        opt_fn = partial(torch.optim.Adagrad)
    elif optimizer_name.lower() == 'adam':
        opt_fn = partial(torch.optim.Adam)
    elif optimizer_name.lower() == 'sgd':
        opt_fn = partial(torch.optim.SGD)
    else:
        raise CustomError("no optimizer selected")

    return opt_fn


def runner(runner_arguments:RunnerArguments):
    """Orchestrates the whole run with given hyper-params"""

    # Setting up logging
    logger.info(f"arguemnts: {locals()}")

    # Setting up seeds for reproducibility and resolving device (cpu/gpu).
    set_seed(runner_arguments.seed)
    device = resolve_device()

    # Create iterator
    iterator_params = {
        'batch_size': runner_arguments.batch_size,
        'lm_encoder_type': 'bert-base-uncased',
        'lm_encoding_to_use': 'use_cls',
        'return_numpy_array': False
    }
    iterators, other_meta_data = generate_data_iterators(dataset_name=runner_arguments.dataset_name, **iterator_params)

    # Create model
    model = get_model(model_name=runner_arguments.model, other_meta_data=other_meta_data, device=device)
    model = model.to(device)

    # Create optimizer
    optimizer = make_opt(model, get_optimizer(runner_arguments.optimizer_name), lr=runner_arguments.lr)

if __name__ == '__main__':
    runner_arguments = RunnerArguments(
        seed = 42,
        dataset_name = 'twitter_hate_speech',
        batch_size = 64,
        model = 'simple_non_linear',
        epochs = 20,
        adv_loss_scale = 0.0,
        save_model_as = 'dummy',
        method = 'unconstrained',
        optimizer_name = 'adam',
        lr = 0.001,
        use_lr_schedule = False
    )

    runner(runner_arguments=runner_arguments)

