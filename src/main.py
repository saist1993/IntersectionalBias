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
from typing import NamedTuple, Dict, Optional
from training_loops import titled_erm_training_loop
from training_loops import adversarial_training_loop
from dataset_iterators import generate_data_iterators
from training_loops import unconstrained_training_loop
from training_loops import adversarial_moe_training_loop
from models import simple_model, adversarial, adversarial_moe
from utils.misc import resolve_device, set_seed, make_opt, CustomError

# Setting up logger


LOG_DIR = Path('../logs')
SAVED_MODEL_PATH = Path('../saved_models/')


class RunnerArguments(NamedTuple):
    """Arguments for the main function"""
    seed: int
    dataset_name: str
    batch_size: int
    model: str
    epochs: int = 20
    save_model_as: Optional[str] = None
    method: str = 'unconstrained'
    optimizer_name: str = 'adam'
    lr: float = 0.001
    use_wandb: bool = False
    adversarial_lambda: float = 0.5
    dataset_size: int = 10000
    attribute_id: Optional[int] = None
    fairness_lambda: float = 0.0
    log_file_name: Optional[str] = None
    fairness_function: str = 'equal_opportunity'
    titled_t:float = 5.0


def get_logger(unique_id_for_run, log_file_name:Optional[str], runner_arguments):
    logger = logging.getLogger(str(unique_id_for_run))

    logger.setLevel('INFO')

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)



    if runner_arguments.log_file_name:
        log_file_name = Path(log_file_name  + ".log")
    else:
        log_file_name = Path( str(unique_id_for_run) + ".log")
    log_file_name = Path(LOG_DIR) / Path(runner_arguments.dataset_name) /  Path(runner_arguments.method) /\
                    Path(runner_arguments.model) / Path(str(runner_arguments.seed)) / Path(str(runner_arguments.fairness_function))/ log_file_name

    log_file_name.parent.mkdir(exist_ok=True, parents=True)

    fileHandler = logging.FileHandler(log_file_name, mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)


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

        if method in ['unconstrained', 'unconstrained_with_fairness_loss', 'tilted_erm']:
            model = simple_model.SimpleNonLinear(model_params)
        elif method == 'adversarial_single':
            total_adv_dim = len(other_meta_data['s_flatten_lookup'])
            model_params['model_arch']['adv'] = {'output_dim': [total_adv_dim]}
            model = adversarial.SimpleNonLinear(model_params)
        elif method in ['adversarial_group', 'adversarial_group_with_fairness_loss']:
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

    # sanity checks - method == unconstrained - fairness_loss = 0.0
    if runner_arguments.method in ['unconstrained', 'adversarial_group']:
        assert runner_arguments.fairness_lambda == 0.0
    # sanity checks - method == unconstrained_with_fairness_loss - fairness_loss > 0.0
    if runner_arguments.method in ['unconstrained_with_fairness_loss', 'adversarial_group_with_fairness_loss']:
        assert runner_arguments.fairness_lambda > 0.0

    # Setting up seeds for reproducibility and resolving device (cpu/gpu).
    set_seed(runner_arguments.seed)
    # device = resolve_device()
    device = torch.device('cpu')

    # setup unique id for the run
    unique_id_for_run = shortuuid.uuid()


    # Setting up logging
    get_logger(unique_id_for_run, runner_arguments.log_file_name, runner_arguments)
    logger = logging.getLogger(str(unique_id_for_run))
    logger.info(f"unique id is:{unique_id_for_run}")

    logger.info(f"arguemnts: {locals()}")


    # Setting up wandb, if need be
    if runner_arguments.use_wandb:
        wandb.init(project="IntersectionalFairness", entity="magnet", config=locals())
        run_id = wandb.run.id
        logging.info(f"wandb run id is {run_id}")



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
        unique_id_for_run = str(unique_id_for_run),
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
                      'attribute_id': runner_arguments.attribute_id,
                      'fairness_lambda': runner_arguments.fairness_lambda,
                      'batch_size': runner_arguments.batch_size,
                      'titled_t': runner_arguments.titled_t,
                      'gamma': 0.2},
        fairness_function=runner_arguments.fairness_function
    )
    # Combine everything

    if 'unconstrained' in runner_arguments.method :
        output = unconstrained_training_loop.training_loop(training_loop_params)
    elif runner_arguments.method in ['adversarial_group', 'adversarial_single', 'adversarial_group_with_fairness_loss']:
        output = adversarial_training_loop.training_loop(training_loop_params)
    elif runner_arguments.method in ['adversarial_moe']:
        output = adversarial_moe_training_loop.training_loop(training_loop_params)
    elif runner_arguments.method == 'tilted_erm':
        output = titled_erm_training_loop.training_loop(training_loop_params)
    else:
        raise NotImplementedError

    output['raw_data'] = other_meta_data['raw_data']
    logging.shutdown()
    return output

if __name__ == '__main__':
    # setting up parser and parsing the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial_lambda', '-adversarial_lambda', help="the lambda in the adv loss equation", type=float,
                        default=1.5)
    parser.add_argument('--fairness_lambda', '-fairness_lambda', help="the lambda in the fairness loss equation", type=float,
                        default=0.0)
    parser.add_argument('--method', '-method', help="unconstrained/adversarial_single/adversarial_group", type=str,
                        default='tilted_erm')
    parser.add_argument('--save_model_as', '-save_model_as', help="unconstrained/adversarial_single/adversarial_group", type=str,
                        default=None)
    parser.add_argument('--dataset_name', '-dataset_name', help="twitter_hate_speech/adult_multi_group",
                        type=str,
                        default='adult_multi_group')

    parser.add_argument('--log_file_name', '-log_file_name', help="the name of the log file",
                        type=str,
                        default='dummy')

    parser.add_argument('--fairness_function', '-fairness_function', help="fairness function to concern with",
                        type=str,
                        default='equal_opportunity')



    args = parser.parse_args()
    if args.save_model_as:
        save_model_as = args.save_model_as + args.dataset_name
    else:
        save_model_as = None


    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)


    runner_arguments = RunnerArguments(
        seed=10,
        dataset_name=args.dataset_name, # twitter_hate_speech
        batch_size=512,
        model='simple_non_linear',
        epochs=200,
        save_model_as=save_model_as,
        method=args.method, # unconstrained, adversarial_single
        optimizer_name='adam',
        lr=0.001,
        use_wandb=False,
        adversarial_lambda=args.adversarial_lambda,
        dataset_size=10000,
        attribute_id=None,  # which attribute to care about!
        fairness_lambda=args.fairness_lambda,
        log_file_name=args.log_file_name,
        fairness_function=args.fairness_function
    )


    output = runner(runner_arguments=runner_arguments)

