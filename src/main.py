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
from training_loops import dro_and_erm
from typing import NamedTuple, Dict, Optional
from training_loops import oracle_with_mixup
from training_loops import fairgrad_training_loop
from training_loops import titled_erm_training_loop
from training_loops import adversarial_training_loop
from dataset_iterators import generate_data_iterators
from training_loops import unconstrained_training_loop
from training_loops import adversarial_moe_training_loop
from models import simple_model, adversarial, adversarial_moe
from fairgrad.torch import CrossEntropyLoss as fairgrad_CrossEntropyLoss
from utils.misc import resolve_device, set_seed, make_opt, CustomError

# Setting up logger


LOG_DIR = Path('../logs')
SAVED_MODEL_PATH = Path('../saved_models/')


# class SimpleModelGenerator(nn.Module):
#     """Fairgrad uses this as complex non linear model"""
#
#     def __init__(self, input_dim):
#         super().__init__()
#
#         self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.25, 0.25, 0.25, 0.25]))
#
#     def forward(self, other_examples):
#         final_output = torch.tensor(0.0, requires_grad=True)
#         for param, group in zip(self.lambda_params, other_examples):
#             x = group['input']
#             final_output = final_output + x*param
#
#         output = {
#             'prediction': final_output,
#             'adv_output': None,
#             'hidden': x,  # just for compatabilit
#             'classifier_hiddens': None,
#             'adv_hiddens': None
#         }
#
#         return output
#
#     @property
#     def layers(self):
#         return torch.nn.ModuleList([self.layer_1, self.layer_2])


class SimpleModelGenerator(nn.Module):
    """Fairgrad uses this as complex non linear model"""

    def __init__(self, input_dim, number_of_params=3):
        super().__init__()

        if number_of_params == 3:
            self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.33, 0.33, 0.33]))
        elif number_of_params == 4:
            self.lambda_params = torch.nn.Parameter(torch.FloatTensor([0.25, 0.25, 0.25, 0.25]))

        self.more_lambda_params = [torch.nn.Parameter(torch.FloatTensor(torch.ones(input_dim))) for i in
                                   range(len(self.lambda_params))]

    def forward(self, other_examples):
        final_output = torch.tensor(0.0, requires_grad=True)
        for param, group, more_params in zip(self.lambda_params, other_examples, self.more_lambda_params):
            x = group['input']
            final_output = final_output + (x*more_params)*param

        output = {
            'prediction': final_output,
            'adv_output': None,
            'hidden': x,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2])

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
    titled_t: float = 5.0
    mixup_rg: float = 0.5
    max_number_of_generated_examples: float = 1.0
    use_dropout: float = 0.2  # 0.0 corresponds to no dropout being applied
    use_batch_norm: float = 0.0  # 0.0 corresponds to no batch norm being applied
    per_group_label_number_of_examples: int = 1000


def get_fairness_related_meta_dict(train_iterator, fairness_measure, fairness_rate, epsilon):
    all_train_y, all_train_s = [], []
    for items in train_iterator:
        all_train_y.append(items['labels'])
        all_train_s.append(items['aux_flattened'])

    all_train_y = np.hstack(all_train_y)
    all_train_s = np.hstack(all_train_s)

    fairness_related_info = {
        'y_train': all_train_y,
        's_train': all_train_s,
        'fairness_measure': fairness_measure,
        'fairness_rate': fairness_rate,
        'epsilon': epsilon
    }

    return fairness_related_info


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


def get_model(method:str, model_name:str, other_meta_data:Dict, device:torch.device, use_batch_norm:float=0.0, use_dropout:float=0.0):
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
            'device': device,
            'use_batch_norm': use_batch_norm,
            'use_dropout': use_dropout
        }

        if method == 'adversarial_single':
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
    #
    # if "fairgrad" in runner_arguments.method:
    #     assert runner_arguments.optimizer_name == 'sgd'
    # else:
    #     assert runner_arguments.optimizer_name == 'adam'

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




    if "only_generated_data" in runner_arguments.method:
        mmd_augmentation_mechanism = "only_generated_data"
    else:
        mmd_augmentation_mechanism = "both_generated_and_real_data"

    # Create iterator
    iterator_params = {
        'batch_size': runner_arguments.batch_size,
        'lm_encoder_type': 'bert-base-uncased',
        'lm_encoding_to_use': 'use_cls',
        'return_numpy_array': True,
        'dataset_size': 1000,
        'max_number_of_generated_examples': runner_arguments.max_number_of_generated_examples,
        'per_group_label_number_of_examples': runner_arguments.per_group_label_number_of_examples,
        'mmd_augmentation_mechanism': mmd_augmentation_mechanism
    }
    iterators, other_meta_data = generate_data_iterators(dataset_name=runner_arguments.dataset_name, **iterator_params)

    other_meta_data['attribute_id'] = runner_arguments.attribute_id

    # Create model
    model = get_model(method=runner_arguments.method, model_name=runner_arguments.model,
                      other_meta_data=other_meta_data,
                      device=device,
                      use_dropout=runner_arguments.use_dropout,
                      use_batch_norm=runner_arguments.use_batch_norm)
    model = model.to(device)


    # Create optimizer and loss function
    optimizer = make_opt(model, get_optimizer(runner_arguments.optimizer_name), lr=runner_arguments.lr)
    if runner_arguments.method == 'fairgrad':
        fairness_related_meta_data = get_fairness_related_meta_dict(iterators[0]['train_iterator'],
                                                                    runner_arguments.fairness_function,
                                                                    fairness_rate=0.01,
                                                                    epsilon=0.0)
        criterion = fairgrad_CrossEntropyLoss(reduction='none', **fairness_related_meta_data)
    else:
        criterion = fairgrad_CrossEntropyLoss(reduction='none')

    # Fairness function (Later)
    # torch.autograd.set_detect_anomaly(True)
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
                      'gamma': 0.2,
                      'mixup_rg': runner_arguments.mixup_rg,
                      'dataset_name': runner_arguments.dataset_name,
                      'seed': runner_arguments.seed,
                      's_to_flattened_s': other_meta_data['s_flatten_lookup'],
                      'dataset_other_meta_data': other_meta_data,
                      'per_group_label_number_of_examples': runner_arguments.per_group_label_number_of_examples,},
        fairness_function=runner_arguments.fairness_function
    )
    # Combine everything

    if 'unconstrained' in runner_arguments.method or  "fairgrad" in runner_arguments.method:
        output = unconstrained_training_loop.training_loop(training_loop_params)
    # elif "fairgrad" in runner_arguments.method:
    #     output = fairgrad_training_loop.orchestrator(training_loop_params)
    elif runner_arguments.method in ['adversarial_group', 'adversarial_single', 'adversarial_group_with_fairness_loss']:
        output = adversarial_training_loop.training_loop(training_loop_params)
    elif runner_arguments.method in ['adversarial_moe']:
        output = adversarial_moe_training_loop.training_loop(training_loop_params)
    elif runner_arguments.method in ['only_titled_erm', 'only_mixup', 'only_mixup_with_loss_group',
                                     'tilted_erm_with_mixup', 'tilted_erm_with_fairness_loss',
                                     'tilted_erm_with_mixup_only_one_group', 'only_mixup_with_abstract_group',
                                     'weighted_sample_erm', 'only_titled_erm_with_weights',
                                     'only_tilted_erm_with_abstract_group', 'tilted_erm_with_mixup_augmentation',
                                     'only_mixup_based_on_distance', 'tilted_erm_with_mixup_based_on_distance',
                                     'only_mixup_based_on_distance_and_augmentation',
                                     'only_tilted_dro',
                                     'only_tilted_erm_with_mixup_augmentation_lambda_weights',
                                     'only_tilted_erm_with_mixup_augmentation_lambda_weights_v2',
                                     'only_tilted_erm_with_mixup_augmentation_lambda_weights_v3',
                                     'only_tilted_erm_with_mixup_augmentation_lambda_weights_v4',
                                     'only_tilted_erm_with_weights_on_loss',
                                     'train_with_mixup_only_one_group_based_distance_v2',
                                     'train_with_mixup_only_one_group_based_distance_v3',
                                     'only_titled_erm_with_mask',
                                     'only_tilted_erm_generic', 'only_tilted_erm_with_mask_on_tpr',
                                     'only_tilted_erm_with_weighted_loss_via_global_weight',
                                     'only_tilted_erm_with_mask_on_tpr_and_weighted_loss_via_global_weight',
                                     'train_only_group_dro',
                                     'train_only_group_dro_with_weighted_sampling',
                                     'only_mixup_based_on_distance_fid',
                                     'train_only_group_dro_with_mixup_with_distance',
                                     'train_only_group_dro_with_mixup_with_random',
                                     'train_only_group_dro_with_mixup_with_random_with_weighted_sampling',
                                     'train_only_group_dro_with_mixup_with_distance_with_weighted_sampling',
                                     'train_only_group_dro_with_augmentation_static_positive_and_negative_weights',
                                     'simple_mixup_data_augmentation',
                                     'lisa_based_mixup',
                                     'lisa_based_mixup_with_mixup_regularizer',
                                     'lisa_based_mixup_with_distance',
                                     'lisa_based_mixup_with_mixup_regularizer_and_with_distance',
                                     'train_only_group_dro_with_mixup_super_group',
                                     'train_only_group_dro_with_mixup_regularizer_super_group',
                                     'train_only_group_dro_with_super_group',
                                     'train_only_group_dro_with_mixup_regularizer_super_group_data_augmentation',
                                     'train_only_group_dro_with_super_group_data_augmentation',
                                     'take_2_train_lisa_based_mixup',
                                     'take_2_train_lisa_based_mixup_with_mixup_regularizer',
                                     'train_only_group_dro_with_data_augmentation_via_mixup_super_group',
                                     'train_only_group_dro_with_data_augmentation_via_mixup_super_group_with_mixup_regularizer',
                                     'train_only_group_dro_with_data_augmentation_via_mixup_super_group_and_example_similarity_v1',
                                     'train_only_group_dro_with_data_augmentation_via_mixup_super_group_and_example_similarity_v2',
                                     'train_only_group_dro_with_mixup_regularizer_super_group_v2',
                                     'erm_super_group_with_simplified_fairness_loss',
                                     'train_only_group_dro_super_group_with_simplified_fairness_loss',
                                     'train_only_group_dro_super_group_with_symmetric_mixup_regularizer',
                                     'train_only_group_dro_super_group_with_symmetric_mixup_regularizer_integrated',
                                     'train_only_group_dro_super_group_with_non_symmetric_mixup_regularizer',
                                     'train_only_group_dro_super_group_with_non_symmetric_mixup_regularizer_integrated',
                                     'train_only_group_dro_with_mixup_regularizer_super_group_and_data_augmentation'
                                     ]:
        output = titled_erm_training_loop.training_loop(training_loop_params)
    elif "oracle" in runner_arguments.method:
        output = oracle_with_mixup.orchestrator(training_loop_parameters=training_loop_params)
    else:
        output = dro_and_erm.orchestrator(training_loop_parameters=training_loop_params)

    output['raw_data'] = other_meta_data['raw_data']
    logging.shutdown()
    return output

if __name__ == '__main__':
    # setting up parser and parsing the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', help="seed for reproduction",
                        type=int,
                        default=50)

    parser.add_argument('--batch_size', '-batch_size', help="seed for reproduction",
                        type=int,
                        default=500)

    parser.add_argument('--epochs', '-epochs', help="epochs to work on",
                        type=int,
                        default=100)

    parser.add_argument('--model', '-model', help="simple_non_linear",
                        type=str,
                        default='simple_non_linear')

    parser.add_argument('--adversarial_lambda', '-adversarial_lambda', help="the lambda in the adv loss equation", type=float,
                        default=0.0)
    parser.add_argument('--fairness_lambda', '-fairness_lambda', help="the lambda in the fairness loss equation", type=float,
                        default=0.0)
    parser.add_argument('--method', '-method', help="unconstrained/adversarial_single/adversarial_group", type=str,
                            default='fairgrad_mixup_generated_and_real_data_only_generated_data')
    parser.add_argument('--save_model_as', '-save_model_as', help="unconstrained/adversarial_single/adversarial_group", type=str,
                        default=None)
    parser.add_argument('--dataset_name', '-dataset_name', help="twitter_hate_speech/adult_multi_group/celeb_multigroup_v3",
                        type=str,
                        default='twitter_hate_speech_augmented')

    parser.add_argument('--log_file_name', '-log_file_name', help="the name of the log file",
                        type=str,
                        default='dummy')

    parser.add_argument('--fairness_function', '-fairness_function', help="fairness function to concern with",
                        type=str,
                        default='equal_odds')

    parser.add_argument('--titled_t', '-titled_t', help="fairness function to concern with",
                        type=float,
                        default=0.5)

    parser.add_argument('--mixup_rg', '-mixup_rg', help="fairness function to concern with",
                        type=float,
                        default=30.0)

    parser.add_argument('--max_number_of_generated_examples', '-max_number_of_generated_examples', help="fairness function to concern with",
                        type=float,
                        default=1.0)

    parser.add_argument('--use_dropout', '-use_dropout',
                        help="dropout(p=use_dropout)",
                        type=float,
                        default=0.1)

    parser.add_argument('--use_batch_norm', '-use_batch_norm',
                        help="batch norm of 0.0 means no norm else batch norm is applied",
                        type=float,
                        default=0.0)

    parser.add_argument('--per_group_label_number_of_examples', '-per_group_label_number_of_examples',
                        help="number of example to generate per group and label = 000+ -> 1000",
                        type=int,
                        default=1000)



    args = parser.parse_args()
    if args.save_model_as:
        save_model_as = args.save_model_as + args.dataset_name
    else:
        save_model_as = None



    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)


    runner_arguments = RunnerArguments(
        seed=args.seed,
        dataset_name=args.dataset_name, # twitter_hate_speech
        batch_size=args.batch_size,
        model=args.model,
        epochs=args.epochs,
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
        fairness_function=args.fairness_function,
        titled_t=args.titled_t,
        mixup_rg=args.mixup_rg,
        max_number_of_generated_examples=args.max_number_of_generated_examples,
        use_dropout=args.use_dropout,
        use_batch_norm=args.use_batch_norm,
        per_group_label_number_of_examples=args.per_group_label_number_of_examples
    )


    output = runner(runner_arguments=runner_arguments)

