import torch
import random
import numpy as np


def make_opt(model, opt_fn: torch.optim, lr: float = 0.001):
    """
        Based on model.layers it creates diff param groups in opt.
    """
    assert hasattr(model, 'layers'), "The model does not have a layers attribute. Check TODO-URL for a how-to"
    return opt_fn([{'params': l.parameters(), 'lr': lr, 'weight_decay': 0.001} for l in model.layers])

class CustomError(Exception): pass


def resolve_device(device = None) -> torch.device:
    """Resolve a torch.device given a desired device (string)."""
    if device is None or device == 'gpu':
        device = 'cuda'
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
        print('No cuda devices were available. The model runs on CPU')
    return device


def set_seed(seed:int) -> None:
    """Sets seed for random, torch, and numpy"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False