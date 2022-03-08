import torch
import random
import logging
import numpy as np
from utils.misc import resolve_device, set_seed
from typing import NamedTuple

# Setting up logger
logger = logging.getLogger(__name__)

class MainArguments(NamedTuple):
    """Arguments for the main function"""
    seed: int
    dataset_name: str
    batch_size: int
    model: str
    epochs: int = 20
    adv_loss_scale: float = 0.0
    save_model_as: str = 'dummy'
    method: str = 'unconstrained'
    optimizer: str = 'adam'
    lr: float = 0.001
    use_lr_schedule: bool = False


def main(main_arguments:MainArguments):
    """Orchestrates the whole run with given hyper-params"""

    # setting up logging
    logger.info(f"arguemnts: {locals()}")

    # setting up seeds for reproducibility
    set_seed(main_arguments.seed)
    device = resolve_device()  # if cuda: then cuda else cpu
