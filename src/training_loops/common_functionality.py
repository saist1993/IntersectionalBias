import os
import copy
import torch
import wandb
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations
from metrics import calculate_epoch_metric
from typing import Dict, Callable, Optional



@dataclass
class TrainParameters:
    model: torch.nn.Module
    iterator: Dict
    optimizer: torch.optim
    criterion: Callable
    device: torch.device
    other_params: Dict
    per_epoch_metric: Callable
    mode: str
    fairness_function:str


@dataclass
class TrainingLoopParameters:
    n_epochs: int
    model: torch.nn.Module
    iterators: Dict
    optimizer: torch.optim
    criterion: Callable
    device: torch.device
    use_wandb: bool
    other_params: Dict
    save_model_as: Optional[str]
    fairness_function: str
    unique_id_for_run: str

def get_fairness_loss(fairness_function, loss, preds, aux, label, all_patterns):

    if fairness_function == 'demographic_parity':
        losses = []
        preds_mask = torch.argmax(preds,1) == 1
        for pattern in all_patterns:
            aux_mask = torch.einsum("ij->i", torch.eq(aux, pattern)) > aux.shape[1] - 1
            final_mask = torch.logical_and(preds_mask, aux_mask)
            _loss = loss[final_mask]
            if len(_loss) > 0:
                losses.append(torch.mean(_loss))
        final_loss = []
        for l1, l2 in combinations(losses, 2):
            final_loss.append(abs(l1-l2))
        if len(final_loss) == 0:
            return None
        # if len(final_loss) <
        return torch.stack(final_loss).sum()
    elif fairness_function == 'equal_odds' or fairness_function == 'equal_opportunity':
        losses = []
        label_mask_1 = label == 1
        label_mask_0 = label == 0
        for pattern in all_patterns:
            aux_mask = torch.einsum("ij->i", torch.eq(aux, pattern)) > aux.shape[1] - 1
            final_mask_1 = torch.logical_and(label_mask_1, aux_mask)
            final_mask_0 = torch.logical_and(label_mask_0, aux_mask)
            _loss_1 = loss[final_mask_1]
            _loss_0 = loss[final_mask_0]
            if len(_loss_1) == 0:
                _loss_1 = None
            else:
                _loss_1 = torch.mean(_loss_1)

            if len(_loss_0) == 0:
                _loss_0 = None
            else:
                _loss_0 = torch.mean(_loss_0)

            losses.append([_loss_0,_loss_1])
        final_loss = []
        for l1, l2 in combinations(losses, 2):
            if l1[0] != None and l2[0] != None and fairness_function == 'equal_odds':
                final_loss.append(abs(l1[0] - l2[0]))
            if l1[1] != None and l2[1] != None:
                final_loss.append(abs(l1[1] - l2[1]))
        if len(final_loss) == 0:
            return None
        return torch.stack(final_loss).sum()
    else:
        raise NotImplementedError


def per_epoch_metric(epoch_output, epoch_input, fairness_function, attribute_id=None):
    """
    :param epoch_output: access all the batch output.
    :param epoch_input: Iterator to access the gold data and inputs
    :return:
    """
    # Step1: Flatten everything for easier analysis
    all_label = []
    all_prediction = []
    all_loss = []
    all_s = []
    all_adversarial_output = []
    all_s_flatten = []

    flag = True # if the output is of the form adversarial_single
    try:
        if type(epoch_output[0]['adv_outputs']) is  list:
            flag = False
    except KeyError:
        flag = False

    # Now there are two variations of s
    # These are batch_output['adv_outputs'] type is list or not a list
    for batch_output, batch_input in zip(epoch_output, epoch_input):
        all_prediction.append(batch_output['prediction'].detach().numpy())
        all_loss.append(batch_output['loss_batch'])
        all_label.append(batch_input['labels'].numpy())
        all_s.append(batch_input['aux'].numpy())
        if attribute_id is not None:
            all_s_flatten.append(batch_input['aux'][:,attribute_id].numpy())
        else:
            all_s_flatten.append(batch_input['aux_flattened'].numpy())
        if flag:
            all_adversarial_output.append(batch_output['adv_outputs'].detach().numpy()) # This line might break!

        # all_s_prediction(batch_output[''])

    try:
        if not flag:
            # this is a adversarial group
            for j in range(len(epoch_output[0]['adv_outputs'])):
                all_adversarial_output.append(torch.vstack([i['adv_outputs'][j] for i in epoch_output]).argmax(1).detach().numpy())
        else:
            all_adversarial_output = np.vstack(all_adversarial_output).argmax(1)
    except KeyError:
        all_adversarial_output = None
    all_prediction = np.vstack(all_prediction).argmax(1)
    all_label = np.hstack(all_label)
    all_s = np.vstack(all_s)
    all_s_flatten = np.hstack(all_s_flatten)


    # Calculate accuracy
    # accuracy = fairness_functions.calculate_accuracy_classification(predictions=all_prediction, labels=all_label)
    #
    # # Calculate fairness
    # accuracy_parity_fairness_metric_tracker, true_positive_rate_fairness_metric_tracker\
    #     = calculate_fairness(prediction=all_prediction, label=all_label, aux=all_s)
    # epoch_metric = EpochMetricTracker(accuracy=accuracy, accuracy_parity=accuracy_parity_fairness_metric_tracker,
    #                                   true_positive_rate=true_positive_rate_fairness_metric_tracker)

    other_meta_data = {
        # 'fairness_mode': ['demographic_parity', 'equal_opportunity', 'equal_odds'],
        'fairness_mode': [fairness_function],
        'no_fairness': False,
        'adversarial_output': all_adversarial_output,
        'aux_flattened': all_s_flatten
    }

    epoch_metric = calculate_epoch_metric.CalculateEpochMetric(all_prediction, all_label, all_s, other_meta_data).run()
    # epoch_metric = None
    return epoch_metric, np.mean(all_loss)


def log_and_plot_data(epoch_metric, loss, train=True):
    if train:
        suffix = "train_"
    else:
        suffix = "test_"

    wandb.log({suffix + "accuracy": epoch_metric.accuracy,
               suffix + "balanced_accuracy": epoch_metric.balanced_accuracy,
               suffix + "loss": loss})



# def find_best_model(output, fairness_measure = 'equal_opportunity', relexation_threshold=0.02):
#     best_fairness_measure = 100
#     best_fairness_index = 0
#     best_valid_accuracy = max([metric.accuracy for metric in output['all_valid_eps_metric']])
#     for index, validation_metric in enumerate(output['all_valid_eps_metric']):
#         if validation_metric.accuracy >= best_valid_accuracy - relexation_threshold:
#             fairness_value = validation_metric.eps_fairness[fairness_measure].intersectional_bootstrap[0]
#             if fairness_value < best_fairness_measure:
#                 best_fairness_measure = fairness_value
#                 best_fairness_index = index
#     return best_fairness_index


def log_epoch_metric(logger, start_message, epoch_metric, epoch_number, loss):
    epoch_metric.loss = loss
    epoch_metric.epoch_number = epoch_number
    logger.info(f"{start_message} epoch metric: {epoch_metric}")


def generate_combinations(s, k =1):
    all_s_combinations = []

    for i in combinations(range(len(s)),k):
        _temp = copy.deepcopy(s)
        for j in i:
            _temp[j] = 'x'
        all_s_combinations.append(tuple(_temp))

    return all_s_combinations


def sample_batch_sen_idx(all_input, all_label, all_aux, all_aux_flatten, batch_size, s):
    """
        This will sample batch size number of elements from input with given s!
    """
    relevant_index = np.random.choice(np.where(all_aux_flatten==s)[0], size=batch_size, replace=True).tolist()
    # THIS IS DIFFERENT. IN ORIGINAL VERSION IT IS REPLACEMENT TRUE
    batch_input = {
        'labels': torch.LongTensor(all_label[relevant_index]),
        'input': torch.FloatTensor(all_input[relevant_index]),
        'aux': torch.LongTensor(all_aux[relevant_index]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[relevant_index])
    }

    return batch_input

def generate_possible_patterns(s0, s1):
    all_s0_combinations, all_s1_combinations = generate_combinations(s0,1), generate_combinations(s1,1)
    all_unique_s0_combination = list(set(all_s0_combinations) - set(all_s1_combinations))
    all_unique_s0_combination.append(tuple(s0))
    all_unique_s1_combination = list(set(all_s1_combinations) - set(all_s0_combinations))
    all_unique_s1_combination.append(tuple(s1))
    return all_unique_s0_combination, all_unique_s1_combination


def generate_mask(all_s, mask_pattern):
    keep_indices = []

    for index, i in enumerate(mask_pattern):
        if i != 'x':
            keep_indices.append(i == all_s[:, index])
        else:
            keep_indices.append(np.ones_like(all_s[:, 0], dtype='bool'))

    mask = np.ones_like(all_s[:, 0], dtype='bool')

    # mask = [np.logical_and(mask, i) for i in keep_indices]

    for i in keep_indices:
        mask = np.logical_and(mask, i)
    return mask



def generate_flat_output_custom(input_iterator, attribute_id=None):
    all_label = []
    all_s = []
    all_s_flatten = []
    all_input = []

    for batch_input in input_iterator:
        all_label.append(batch_input['labels'].numpy())
        all_s.append(batch_input['aux'].numpy())
        all_s_flatten.append(batch_input['aux_flattened'].numpy())
        all_input.append(batch_input['input'].numpy())

    all_label = np.hstack(all_label)
    all_s = np.vstack(all_s)
    all_s_flatten = np.hstack(all_s_flatten)
    all_input = np.vstack(all_input)

    return all_label, all_s, all_s_flatten, all_input




def training_loop_common(training_loop_parameters: TrainingLoopParameters, train_function):
    logger = logging.getLogger(training_loop_parameters.unique_id_for_run)


    output = {}
    all_train_eps_metrics = []
    all_test_eps_metrics = []
    all_valid_eps_metrics = []
    # models = []
    # best_test_accuracy = 0.0
    # best_eopp = 1.0





    for ep in range(training_loop_parameters.n_epochs):

        # Train split
        logger.info("start of epoch block  ")
        train_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['train_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='train',
            fairness_function=training_loop_parameters.fairness_function)

        train_epoch_metric, loss = train_function(train_parameters)
        log_epoch_metric(logger, start_message='train', epoch_metric=train_epoch_metric, epoch_number=ep, loss=loss)


        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=train_epoch_metric, loss=loss, train=True)
        all_train_eps_metrics.append(train_epoch_metric)

        # Valid split
        valid_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['valid_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='evaluate',
            fairness_function=training_loop_parameters.fairness_function)

        valid_epoch_metric, loss = train_function(valid_parameters)
        log_epoch_metric(logger, start_message='valid', epoch_metric=valid_epoch_metric, epoch_number=ep, loss=loss)

        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=valid_epoch_metric, loss=loss, train=True)
        all_valid_eps_metrics.append(train_epoch_metric)


        # test split
        test_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['test_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='evaluate',
            fairness_function=training_loop_parameters.fairness_function)

        test_epoch_metric, loss = train_function(test_parameters)
        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=test_epoch_metric, loss=loss, train=False)
        log_epoch_metric(logger, start_message='test',  epoch_metric=test_epoch_metric, epoch_number=ep, loss=loss)
        all_test_eps_metrics.append(test_epoch_metric)

        logger.info("end of epoch block")

        # models.append(copy.deepcopy(training_loop_parameters.model))

        # if test_epoch_metric.eps_fairness['equal_odds'].intersectional_bootstrap[0] < 1.4:
        #     method = training_loop_parameters.other_params['method']
        #     dataset_name = training_loop_parameters.other_params['dataset_name']
        #     seed = training_loop_parameters.other_params['seed']
        #     _dir = f'../saved_models/{dataset_name}/{method}/{seed}'
        #     Path(_dir).mkdir(parents=True, exist_ok=True)
        #     if training_loop_parameters.save_model_as != None:
        #         torch.save(training_loop_parameters.model.state_dict(),
        #                    f'{_dir}/{training_loop_parameters.fairness_function}_{training_loop_parameters.save_model_as}.pt')
        #
        #     raise IOError


    # Saving the last epoch model.


    output['all_train_eps_metric'] = all_train_eps_metrics
    output['all_test_eps_metric'] = all_test_eps_metrics
    output['all_valid_eps_metric'] = all_valid_eps_metrics
    # index = find_best_model(output, training_loop_parameters.fairness_function)
    # output['best_model_index'] = index
    # method = training_loop_parameters.other_params['method']
    # dataset_name = training_loop_parameters.other_params['dataset_name']
    # seed = training_loop_parameters.other_params['seed']
    # _dir = f'../saved_models/{dataset_name}/{method}/{seed}'
    # Path(_dir).mkdir(parents=True, exist_ok=True)
    # if training_loop_parameters.save_model_as != None:
    #     torch.save(models[index].state_dict(),
    #                f'{_dir}/{training_loop_parameters.fairness_function}_{training_loop_parameters.save_model_as}.pt')


    return output