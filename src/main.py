import torch
import wandb
import config
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
from functools import partial
from models import simple_model, adversarial
from typing import NamedTuple, Dict
from utils import plot_and_visualize
from dataset_iterators import generate_data_iterators
from training_loops import adversarial_training_loop
from training_loops import unconstrained_training_loop
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
    save_model_as: str = 'dummy'
    method: str = 'unconstrained'
    optimizer_name: str = 'adam'
    lr: float = 0.001
    use_lr_schedule: bool = False
    use_wandb: bool = False
    adversarial_lambda: float = 0.5
    dataset_size: int = 10000

def get_model(method:str, model_name:str, other_meta_data:Dict, device:torch.device):
    number_of_aux_label_per_attribute = other_meta_data['number_of_aux_label_per_attribute']
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

        if method == 'unconstrained':
            model = simple_model.SimpleNonLinear(model_params)
        elif method == 'adversarial_single':
            total_adv_dim = len(other_meta_data['s_flatten_lookup'])
            model_params['model_arch']['adv'] = {'output_dim': [total_adv_dim]}
            model = adversarial.SimpleNonLinear(model_params)
        elif method == 'adversarial_group':
            model_params['model_arch']['adv'] = {'output_dim': number_of_aux_label_per_attribute}
            model = adversarial.SimpleNonLinear(model_params)
        elif method == 'adversarial_intersectional':
            pass
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
    """
     Orchestrates the whole run with given hyper-params

     - Assume we have 2 classes and 2 attributes (A1, A2). A1 has two sensitive attribute, while A2 has three sensitive
     attribute

     method - unconstrained - as the name suggest does not impose any fairness constraints.
     method - adversarial_single - use one adversary - adv_dim - 6 classes
     method - adversarial_group - 2 adversary with one adversary having adv_dim as 2 and other as 3
     method - adversarial_intersectional - 6 adversarial with each adversarial predicting the presence of group!
     """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    # Setting up logging
    logger.info(f"arguemnts: {locals()}")

    # Setting up wandb, if need be
    if runner_arguments.use_wandb:
        wandb.init(project="IntersectionalFairness", entity="magnet", config=locals())
        run_id = wandb.run.id
        logger.info(f"wandb run id is {run_id}")


    # Setting up seeds for reproducibility and resolving device (cpu/gpu).
    set_seed(runner_arguments.seed)
    device = resolve_device()


    # Create iterator
    iterator_params = {
        'batch_size': runner_arguments.batch_size,
        'lm_encoder_type': 'bert-base-uncased',
        'lm_encoding_to_use': 'use_cls',
        'return_numpy_array': True,
        'dataset_size': 1000
    }
    iterators, other_meta_data = generate_data_iterators(dataset_name=runner_arguments.dataset_name, **iterator_params)

    # Create model
    model = get_model(method=runner_arguments.method, model_name=runner_arguments.model, other_meta_data=other_meta_data, device=device)
    model = model.to(device)

    # Create optimizer and loss function
    optimizer = make_opt(model, get_optimizer(runner_arguments.optimizer_name), lr=runner_arguments.lr)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Fairness function (Later)

    # Training Loops
    training_loop_params = unconstrained_training_loop.TrainingLoopParameters(
        n_epochs=runner_arguments.epochs,
        model=model,
        iterators=iterators,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_model_as=runner_arguments.save_model_as,
        use_wandb=runner_arguments.use_wandb,
        other_params={'method': runner_arguments.method,
                      'adversarial_lambda': runner_arguments.adversarial_lambda}
    )
    # Combine everything

    if runner_arguments.method == 'unconstrained':
        output = unconstrained_training_loop.training_loop(training_loop_params)
    elif runner_arguments.method in ['adversarial_group', 'adversarial_single']:
        output = adversarial_training_loop.training_loop(training_loop_params)
    else:
        raise NotImplementedError

    output['raw_data'] = other_meta_data['raw_data']

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial_lambda', '-adversarial_lambda', help="the lambda in the adv loss equation", type=float,
                        default=0.5)
    parser.add_argument('--method', '-method', help="unconstrained/adversarial_single/adversarial_group", type=str,
                        default='unconstrained')

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    args = parser.parse_args()


    runner_arguments = RunnerArguments(
        seed=42,
        dataset_name='gaussian_toy', # twitter_hate_speech
        batch_size=512,
        model='simple_non_linear',
        epochs=100,
        save_model_as='dummy',
        method=args.method, # unconstrained, adversarial_single
        optimizer_name='adam',
        lr=0.001,
        use_lr_schedule=False,
        use_wandb=False,
        adversarial_lambda=args.adversarial_lambda,
        dataset_size=10000
    )

    output = runner(runner_arguments=runner_arguments)

    plot_and_visualize.plot_eps_fairness_metric(output['all_train_eps_metric'], "Train ", runner_arguments.use_wandb)
    plot_and_visualize.plot_eps_fairness_metric(output['all_test_eps_metric'], "Test ", runner_arguments.use_wandb)

