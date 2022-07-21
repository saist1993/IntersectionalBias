import torch
import wandb
import config
import random
import logging
import argparse
import shortuuid
import numpy as np
import torch.nn as nn
from pathlib import Path
from functools import partial
from utils import plot_and_visualize
from models import simple_model, adversarial, adversarial_moe
from typing import NamedTuple, Dict, Optional
from dataset_iterators import generate_data_iterators
from training_loops import adversarial_training_loop
from training_loops import unconstrained_training_loop
from training_loops import adversarial_moe_training_loop
from utils.misc import resolve_device, set_seed, make_opt, CustomError

# Setting up logger
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()



consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


LOG_DIR = Path('../logs')
SAVED_MODEL_PATH = Path('../saved_models/')


class RunnerArguments(NamedTuple):
    """Arguments for the main function"""
    seed: int
    dataset_name: str
    batch_size: int
    model: str
    epochs: int = 20
    save_model_as: str = None
    method: str = 'unconstrained'
    optimizer_name: str = 'adam'
    lr: float = 0.001
    use_lr_schedule: bool = False
    use_wandb: bool = False
    adversarial_lambda: float = 0.5
    dataset_size: int = 10000
    attribute_id: Optional[int] = None
    fairness_lambda: float = 0.0
    log_file_name: Optional[str] = None


def get_model(method:str, model_name:str, other_meta_data:Dict, device:torch.device):
    number_of_aux_label_per_attribute = other_meta_data['number_of_aux_label_per_attribute']
    attribute_id = other_meta_data['attribute_id']
    if attribute_id is not None:
        number_of_aux_label_per_attribute = [number_of_aux_label_per_attribute[attribute_id]]

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
        elif method == 'adversarial_moe':
            model_params['model_arch']['adv'] = {'output_dim': number_of_aux_label_per_attribute}
            model = adversarial_moe.SimpleNonLinear(model_params)
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

    # Setting up logging
    unique_id_for_run = shortuuid.uuid()
    if not runner_arguments.log_file_name:
        log_file_name = Path(str(unique_id_for_run))
    else:
        log_file_name = Path(args.log_file_name + "_" +  str(unique_id_for_run))
    fileHandler = logging.FileHandler(LOG_DIR/log_file_name)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

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

    other_meta_data['attribute_id'] = runner_arguments.attribute_id

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
                      'adversarial_lambda': runner_arguments.adversarial_lambda,
                      'attribute_id': runner_arguments.attribute_id}
    )
    # Combine everything

    if runner_arguments.method == 'unconstrained':
        output = unconstrained_training_loop.training_loop(training_loop_params)
    elif runner_arguments.method in ['adversarial_group', 'adversarial_single']:
        output = adversarial_training_loop.training_loop(training_loop_params)
    elif runner_arguments.method in ['adversarial_moe']:
        output = adversarial_moe_training_loop.training_loop(training_loop_params)
    else:
        raise NotImplementedError

    output['raw_data'] = other_meta_data['raw_data']

    return output

if __name__ == '__main__':
    # setting up parser and parsing the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial_lambda', '-adversarial_lambda', help="the lambda in the adv loss equation", type=float,
                        default=0.5)
    parser.add_argument('--fairness_lambda', '-fairness_lambda', help="the lambda in the fairness loss equation", type=float,
                        default=0.0)
    parser.add_argument('--method', '-method', help="unconstrained/adversarial_single/adversarial_group", type=str,
                        default='adversarial_moe')
    parser.add_argument('--save_model_as', '-save_model_as', help="unconstrained/adversarial_single/adversarial_group", type=str,
                        default='adversarial_moe_0.05_0.5')
    parser.add_argument('--dataset_name', '-dataset_name', help="twitter_hate_speech/adult_multi_group",
                        type=str,
                        default='adult_multi_group')
    args = parser.parse_args()
    if args.save_model_as:
        save_model_as = args.save_model_as + args.dataset_name
    else:
        save_model_as = None


    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)


    runner_arguments = RunnerArguments(
        seed=42,
        dataset_name=args.dataset_name, # twitter_hate_speech
        batch_size=1024,
        model='simple_non_linear',
        epochs=150,
        save_model_as=save_model_as,
        method=args.method, # unconstrained, adversarial_single
        optimizer_name='adam',
        lr=0.001,
        use_lr_schedule=False,
        use_wandb=False,
        adversarial_lambda=args.adversarial_lambda,
        dataset_size=10000,
        attribute_id=None,  # which attribute to care about!
        fairness_lambda=args.fairness_lambda
    )

    output = runner(runner_arguments=runner_arguments)

