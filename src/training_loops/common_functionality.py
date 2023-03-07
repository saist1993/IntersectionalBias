import os
import copy
import torch
import wandb
import logging
import numpy as np
from pathlib import Path
from numpy.random import beta
from dataclasses import dataclass
from itertools import combinations
from metrics import calculate_epoch_metric
from typing import Dict, Callable, Optional
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


from pprint import pprint
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
    iterator_set: Optional[Dict]


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


def per_epoch_metric(epoch_output, epoch_input, fairness_function, loss_function=None, attribute_id=None):
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

    group_metrics = {}

    for group in np.unique(all_s, axis=0):
        mask = generate_mask(all_s, group)
        positive_label_mask = all_label == 1
        negative_label_mask = all_label == 0

        positive_mask = np.logical_and(mask, positive_label_mask)
        negative_mask = np.logical_and(mask, negative_label_mask)

        total_group_size = np.sum(mask)
        positive_size = np.sum(positive_mask)
        negative_size = np.sum(negative_mask)

        positive_accuracy = round(accuracy_score(y_true=all_label[positive_mask], y_pred=all_prediction[positive_mask]),
                                  3)
        negative_accuracy = round(accuracy_score(y_true=all_label[negative_mask], y_pred=all_prediction[negative_mask]),
                                  3)
        total_accuracy = round(accuracy_score(y_true=all_label[mask], y_pred=all_prediction[mask]), 3)

        group_metrics[tuple(group)] = [total_group_size, positive_size, negative_size, total_accuracy,
                                       positive_accuracy, negative_accuracy]

        if loss_function:
            total_loss = loss_function(torch.FloatTensor(all_label[mask]), torch.FloatTensor(all_prediction[mask]))/total_group_size
            positive_loss = loss_function(torch.FloatTensor(all_label[positive_mask]), torch.FloatTensor(all_prediction[positive_mask]))/positive_size
            negative_loss = loss_function(torch.FloatTensor(all_label[negative_mask]), torch.FloatTensor(all_prediction[negative_mask]))/negative_size
            group_metrics[tuple(group)].append([total_loss, positive_loss, negative_loss])



    print(group_metrics)

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
    epoch_metric.other_info = group_metrics
    return epoch_metric, np.mean(all_loss)


def log_and_plot_data(epoch_metric, loss, train=True):
    if train:
        suffix = "train_"
    else:
        suffix = "test_"

    wandb.log({suffix + "accuracy": epoch_metric.accuracy,
               suffix + "balanced_accuracy": epoch_metric.balanced_accuracy,
               suffix + "loss": loss})



def find_best_model(output, fairness_measure = 'equal_opportunity', relexation_threshold=0.02):
    best_fairness_measure = 100
    best_fairness_index = 0
    best_valid_accuracy = max([metric.accuracy for metric in output['all_valid_eps_metric']])
    for index, validation_metric in enumerate(output['all_valid_eps_metric']):
        if validation_metric.accuracy >= best_valid_accuracy - relexation_threshold:
            fairness_value = validation_metric.eps_fairness[fairness_measure].intersectional_bootstrap[0]
            if fairness_value < best_fairness_measure:
                best_fairness_measure = fairness_value
                best_fairness_index = index
    return best_fairness_index


def log_epoch_metric(logger, start_message, epoch_metric, epoch_number, loss):
    epoch_metric.loss = loss
    epoch_metric.epoch_number = epoch_number
    logger.info(f"{start_message} epoch metric: {epoch_metric}")


def generate_combinations(s, k =1):
    all_s_combinations = []

    for i in combinations(range(len(s)),k):
        _temp = list(copy.deepcopy(s))
        for j in i:
            _temp[j] = 'x'
        all_s_combinations.append(tuple(_temp))

    return all_s_combinations

#
# def generate_combinations(s, k=1):
#     def flip_bit(bit):
#         if bit == 1:
#             return 0
#         else:
#             return 1
#
#     all_s_combinations = []
#
#     for index, i in enumerate(s):
#         temp = list(copy.deepcopy(s))
#         temp[index] = flip_bit(i)
#         all_s_combinations.append(tuple(temp))
#
#     return all_s_combinations



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



def randomly_sample_data(train_tilted_params, s_group_0, s_group_1):
    items_group_0 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                        train_tilted_params.other_params['all_label'],
                                        train_tilted_params.other_params['all_aux'],
                                        train_tilted_params.other_params['all_aux_flatten'],
                                        train_tilted_params.other_params['batch_size'],
                                        s_group_0)

    if s_group_1 is not None:

        items_group_1 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                             train_tilted_params.other_params['all_label'],
                                             train_tilted_params.other_params['all_aux'],
                                             train_tilted_params.other_params['all_aux_flatten'],
                                             train_tilted_params.other_params['batch_size'],
                                             s_group_1)
    else:
        items_group_1 = None

    return items_group_0, items_group_1


def sample_batch_sen_idx_with_y_abstract(all_input, all_label, all_aux, all_aux_flatten, batch_size, s0, s1):
    """
        This will sample batch size number of elements from input with given s!
    """

    all_unique_s0_combination, all_unique_s1_combination = generate_possible_patterns(s0, s1)
    mask_s0 = np.logical_or.reduce([generate_mask(all_aux, mask_pattern) for mask_pattern in all_unique_s0_combination])
    mask_s1 = np.logical_or.reduce([generate_mask(all_aux, mask_pattern) for mask_pattern in all_unique_s1_combination])

    index_s0_0 = np.where(np.logical_and(mask_s0, all_label==0) == True)[0]
    index_s0_1 = np.where(np.logical_and(mask_s0, all_label==1) == True)[0]

    index_s1_0 = np.where(np.logical_and(mask_s1, all_label == 0) == True)[0]
    index_s1_1 = np.where(np.logical_and(mask_s1, all_label == 1) == True)[0]


    relevant_index_s0 = np.random.choice(index_s0_0, size=int(batch_size/2), replace=True).tolist()
    relevant_index_s0 = relevant_index_s0 + np.random.choice(index_s0_1,size=int(batch_size/2), replace=True).tolist()

    relevant_index_s1 = np.random.choice(index_s1_0,size=int(batch_size/2), replace=True).tolist()
    relevant_index_s1 = relevant_index_s1 + np.random.choice(index_s1_1,size=int(batch_size/2), replace=True).tolist()

    # THIS IS DIFFERENT. IN ORIGINAL VERSION IT IS REPLACEMENT TRUE
    batch_input_s0 = {
        'labels': torch.LongTensor(all_label[relevant_index_s0]),
        'input': torch.FloatTensor(all_input[relevant_index_s0]),
        'aux': torch.LongTensor(all_aux[relevant_index_s0]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[relevant_index_s0])
    }

    batch_input_s1 = {
        'labels': torch.LongTensor(all_label[relevant_index_s1]),
        'input': torch.FloatTensor(all_input[relevant_index_s1]),
        'aux': torch.LongTensor(all_aux[relevant_index_s1]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[relevant_index_s1])
    }

    return batch_input_s0, batch_input_s1



def sample_batch_sen_idx_with_y(all_input, all_label, all_aux, all_aux_flatten, batch_size, s):
    """
        This will sample batch size number of elements from input with given s!
    """
    index_s_0 = np.where(np.logical_and(all_aux_flatten==s, all_label==0) == True)[0]
    index_s_1 = np.where(np.logical_and(all_aux_flatten==s, all_label==1) == True)[0]
    relevant_index = np.random.choice(index_s_0, size=int(batch_size/2), replace=True).tolist()
    relevant_index = relevant_index + np.random.choice(index_s_1,size=int(batch_size/2), replace=True).tolist()

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



def mixup_sub_routine_original(train_tilted_params:TrainParameters, items_group_0, items_group_1, model, gamma=None):
    alpha = 1.0
    if not gamma:
        gamma = beta(alpha, alpha)

    if train_tilted_params.fairness_function == 'demographic_parity':
        batch_x_mix = items_group_0['input'] * gamma + items_group_1['input'] * (1 - gamma)
        batch_x_mix = batch_x_mix.requires_grad_(True)
        output_mixup = model({'input': batch_x_mix})
        gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[
            0]  # may be .sum()

        batch_x_d = items_group_1['input'] - items_group_0['input']
        grad_inn = (gradx * batch_x_d).sum(1)
        E_grad = grad_inn.mean(0)
        loss_reg = torch.abs(E_grad)

    elif train_tilted_params.fairness_function == 'equal_odds' or \
            train_tilted_params.fairness_function == 'equal_opportunity':
        split_index = int(train_tilted_params.other_params['batch_size'] / 2)
        if train_tilted_params.fairness_function == 'equal_odds':
            gold_labels = [0, 1]
        elif train_tilted_params.fairness_function == 'equal_opportunity':
            gold_labels = [1]
        else:
            raise NotImplementedError
        loss_reg = 0
        for i in gold_labels:
            if i == 0:
                index_start = 0
                index_end = split_index
            elif i == 1:
                index_start = split_index
                index_end = -1
            else:
                raise NotImplementedError("only support binary labels!")

            batch_x_mix = items_group_0['input'][index_start:index_end] * gamma + items_group_1['input'][
                                                                                  index_start:index_end] * (
                                  1 - gamma)    # this is the point wise addition which forces this equal 1s,0s representation
            batch_x_mix = batch_x_mix.requires_grad_(True)
            output_mixup = model({'input': batch_x_mix})
            gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[
                0]  # may be .sum()

            batch_x_d = items_group_1['input'][index_start:index_end] - items_group_0['input'][
                                                                        index_start:index_end]
            grad_inn = (gradx * batch_x_d).sum(1)
            E_grad = grad_inn.mean(0)
            loss_reg = loss_reg + torch.abs(E_grad)

    else:
        raise NotImplementedError


    return loss_reg


def mixup_sub_routine(train_tilted_params:TrainParameters, items_group_0, items_group_1, model, gamma=None):
    alpha = 1.0
    gamma = beta(alpha, alpha)

    if train_tilted_params.fairness_function == 'demographic_parity':
        batch_x_mix = items_group_0['input'] * gamma + items_group_1['input'] * (1 - gamma)
        batch_x_mix = batch_x_mix.requires_grad_(True)
        output_mixup = model({'input': batch_x_mix})
        gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[
            0]  # may be .sum()

        batch_x_d = items_group_1['input'] - items_group_0['input']
        grad_inn = (gradx * batch_x_d).sum(1)
        E_grad = grad_inn.mean(0)
        if train_tilted_params.other_params['method'] == 'only_mixup_with_loss_group':
            loss_reg = torch.abs(E_grad) / torch.mean(loss[len(items_group_0['input']):])
        else:
            loss_reg = torch.abs(E_grad)

    elif train_tilted_params.fairness_function == 'equal_odds' or \
            train_tilted_params.fairness_function == 'equal_opportunity':
        split_index = int(train_tilted_params.other_params['batch_size'] / 2)
        if train_tilted_params.fairness_function == 'equal_odds':
            gold_labels = [0, 1]
        elif train_tilted_params.fairness_function == 'equal_opportunity':
            gold_labels = [1]
        else:
            raise NotImplementedError
        loss_reg = 0
        for i in gold_labels:
            if i == 0:
                index_start = 0
                index_end = split_index
            elif i == 1:
                index_start = split_index
                index_end = -1
            else:
                raise NotImplementedError("only support binary labels!")

            batch_x_mix = items_group_0['input'][index_start:index_end] * gamma + items_group_1['input'][
                                                                                  index_start:index_end] * (
                                  1 - gamma)
            batch_x_mix = batch_x_mix.requires_grad_(True)
            output_mixup = model({'input': batch_x_mix})
            gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[
                0]  # may be .sum()

            batch_x_d = items_group_1['input'][index_start:index_end] - items_group_0['input'][
                                                                        index_start:index_end]
            grad_inn = (gradx * batch_x_d).sum(1)
            E_grad = grad_inn.mean(0)
            if train_tilted_params.other_params['method'] == 'only_mixup_with_loss_group':
                loss_reg = loss_reg + torch.abs(E_grad) / torch.mean(loss[index_start:index_end])
            else:
                loss_reg = loss_reg + torch.abs(E_grad)
            # loss_reg = loss_reg + torch.abs(E_grad)

    else:
        raise NotImplementedError

    return loss_reg

def sample_data(train_tilted_params, s_group_0, s_group_1):
    if train_tilted_params.fairness_function == 'demographic_parity':
        items_group_0 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                             train_tilted_params.other_params['all_label'],
                                             train_tilted_params.other_params['all_aux'],
                                             train_tilted_params.other_params['all_aux_flatten'],
                                             train_tilted_params.other_params['batch_size'],
                                             s_group_0)

        if s_group_1 is not None:

            items_group_1 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                                 train_tilted_params.other_params['all_label'],
                                                 train_tilted_params.other_params['all_aux'],
                                                 train_tilted_params.other_params['all_aux_flatten'],
                                                 train_tilted_params.other_params['batch_size'],
                                                 s_group_1)
        else:
            items_group_1 = None

    elif train_tilted_params.fairness_function == 'equal_odds' or \
            train_tilted_params.fairness_function == 'equal_opportunity':
        # group splits -
        # What we want is y=0,g=g0 and y=1,g=g0
        # here items_group_0 say with batch 500 -> first 250 are 0 label and next (last) 250 are 1 label
        items_group_0 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                                   train_tilted_params.other_params['all_label'],
                                                                   train_tilted_params.other_params['all_aux'],
                                                                   train_tilted_params.other_params['all_aux_flatten'],
                                                                   train_tilted_params.other_params['batch_size'],
                                                                   s_group_0)

        if s_group_1 is not None:
            items_group_1 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                                       train_tilted_params.other_params['all_label'],
                                                                       train_tilted_params.other_params['all_aux'],
                                                                       train_tilted_params.other_params['all_aux_flatten'],
                                                                       train_tilted_params.other_params['batch_size'],
                                                                       s_group_1)
        else:
            items_group_1 = None
        # group split

        # class split

    else:
        raise NotImplementedError

    for key in items_group_0.keys():
        items_group_0[key] = items_group_0[key].to(train_tilted_params.device)

    if items_group_1 is not None:

        for key in items_group_1.keys():
            items_group_1[key] = items_group_1[key].to(train_tilted_params.device)

    return items_group_0, items_group_1


def augment_current_data_via_mixup(train_tilted_params, s_group_0, s_group_1, items_group_0, items_group_1, epoch_number=0):
     # 'input': torch.FloatTensor(all_input[relevant_index])

    def custom_routine(s, items):
        augmented_X, augmented_s, augmented_y, augmented_s_flat =\
            train_tilted_params.other_params['all_input_augmented'], train_tilted_params.other_params['all_aux_augmented'],\
                train_tilted_params.other_params['all_label_augmented'],\
                train_tilted_params.other_params['all_aux_flatten_augmented']

        augmented_s_flat = np.asarray(augmented_s_flat)

        size_of_data = int(len(items['input'])/2)
        index_0 = np.where(np.logical_and(augmented_s_flat==s, augmented_y==0) == True)[0]
        index_1 = np.where(np.logical_and(augmented_s_flat==s, augmented_y==1) == True)[0]

        length_of_index_0 = len(index_0)
        length_of_index_1 = len(index_1)

        # if epoch_number < 5:
        #     index_0 = index_0[:int(length_of_index_0/3)]
        #     index_1 = index_1[:int(length_of_index_1/3)]
        # elif epoch_number > 5 and epoch_number < 10:
        #     index_0 = index_0[int(length_of_index_0 / 3): 2*int(length_of_index_0 / 3)]
        #     index_1 = index_1[int(length_of_index_1 / 3): 2*int(length_of_index_0 / 3)]
        # else:
        #     index_0 = index_0[2 * int(length_of_index_0 / 3):]
        #     index_1 = index_1[2 * int(length_of_index_0 / 3):]

        # index_0 = index_0[:int((epoch_number+1)*100000)]
        # index_1 = index_1[:int((epoch_number+1)*100000)]


        relevant_index = np.random.choice(index_0, size=size_of_data, replace=True).tolist()
        relevant_index = relevant_index + np.random.choice(index_1, size=size_of_data, replace=True).tolist()

        relevent_augmented_X = augmented_X[relevant_index]
        relevent_augmented_y = augmented_y[relevant_index]

        alpha = 1.0
        gamma = beta(alpha, alpha)

        shuffle_input = torch.randperm(len(items['input']))
        items['input'] = items['input'][shuffle_input]
        items['labels'] = items['labels'][shuffle_input]

        shuffle_generated_data = torch.randperm(len(relevant_index))
        relevent_augmented_X = relevent_augmented_X[shuffle_generated_data]
        relevent_augmented_y  = relevent_augmented_y [shuffle_generated_data]



        if gamma > (1-gamma):
            input_x_mix = items['input']*gamma + torch.FloatTensor(relevent_augmented_X)*(1-gamma)

            input_label_mix = torch.nn.functional.one_hot(items['labels'])*gamma + torch.nn.functional.one_hot(torch.tensor(relevent_augmented_y).to(torch.int64))*(1-gamma)
        else:
            # input_x_mix = items['input'] * (1-gamma) + torch.FloatTensor(relevent_augmented_X) * gamma
            # input_label_mix = torch.nn.functional.one_hot(items['labels']) * (1 - gamma) +  torch.nn.functional.one_hot(torch.tensor(relevent_augmented_y).to(torch.int64))*gamma

            input_x_mix = items['input'] * gamma + torch.FloatTensor(relevent_augmented_X) * (1 - gamma)

            input_label_mix = torch.nn.functional.one_hot(items['labels']) * gamma + torch.nn.functional.one_hot(
                torch.tensor(relevent_augmented_y).to(torch.int64)) * (1 - gamma)



        # input_x_mix = items['input'] * gamma + torch.FloatTensor(relevent_augmented_X) * (1 - gamma)

        items['input'] = input_x_mix
        items['original_labels'] = items['labels']
        items['labels'] = input_label_mix

        return items

    if s_group_0 is not None:
        items_group_0 = custom_routine(s_group_0, items_group_0)

    if s_group_1 is not None:
        items_group_1 = custom_routine(s_group_1, items_group_1)

    return items_group_0, items_group_1


def augmented_sampling(train_tilted_params, s_group_0, s_group_1):
    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}
    input_group_0, input_group_1 = sample_data(train_tilted_params, s_group_0, s_group_1)
    group_weight_lambda = train_tilted_params.other_params['group_to_lambda_weight']

    augmented_input_group_0, _ = sample_batch_sen_idx_with_augmentation_with_lambda_custom_with_positive_and_negative_seperate(
        train_tilted_params.other_params['all_input'],
        train_tilted_params.other_params['all_label'],
        train_tilted_params.other_params['all_aux'],
        train_tilted_params.other_params['all_aux_flatten'],
        train_tilted_params.other_params['batch_size'],
        flattened_s_to_s[s_group_0],
        group_weight_lambda[s_group_0],
        train_tilted_params.other_params['scalar'],
        input_group_0
    )

    augmented_input_group_1, _ = sample_batch_sen_idx_with_augmentation_with_lambda_custom_with_positive_and_negative_seperate(
        train_tilted_params.other_params['all_input'],
        train_tilted_params.other_params['all_label'],
        train_tilted_params.other_params['all_aux'],
        train_tilted_params.other_params['all_aux_flatten'],
        train_tilted_params.other_params['batch_size'],
        flattened_s_to_s[s_group_1],
        group_weight_lambda[s_group_1],
        train_tilted_params.other_params['scalar'],
        input_group_1
    )

    return augmented_input_group_0, augmented_input_group_1

def sample_batch_sen_idx_with_augmentation_with_lambda_custom_with_positive_and_negative_seperate\
                (all_input, all_label, all_aux, all_aux_flatten, batch_size, s, group_weight_lambda, scalar, input_s):
    all_extra_combinations = generate_combinations(s, k=1)
    extra_mask_s = [generate_mask(all_aux, mask_pattern) for mask_pattern in all_extra_combinations] # this stores all the combinations
    mask_s = generate_mask(all_aux,s)

    # positive and negative examples in input_s
    number_of_positive_examples = torch.sum(input_s['labels']).item()
    number_of_negative_examples = input_s['labels'].shape[0] - number_of_positive_examples

    all_group_examples_positive = []
    all_group_examples_negative = []

    for mask in extra_mask_s:
        mask_positive = np.logical_and(all_label == 1, mask == True)
        mask_negative = np.logical_and(all_label == 0, mask == True)
        positive_examples =  np.random.choice(np.where(mask_positive == True)[0], size=number_of_positive_examples, replace=True) # find somehow only positive example
        negative_examples =  np.random.choice(np.where(mask_negative == True)[0], size=number_of_negative_examples, replace=True) # find somehow only positive example
        all_group_examples_positive.append(positive_examples)
        all_group_examples_negative.append(negative_examples)
        # stack them
    # now comes the augmentations


    try:
        augmented_input_positive =  torch.FloatTensor(np.sum([lambda_weight[0] * all_input[group] for group, lambda_weight in zip(all_group_examples_positive, group_weight_lambda)],axis=0))
    except IndexError:
        print("here")
    augmented_input_negative =  torch.FloatTensor(np.sum([lambda_weight[1] * all_input[group] for group, lambda_weight in zip(all_group_examples_negative, group_weight_lambda)],axis=0))

    augmented_output_positive = [torch.LongTensor(all_label[group]) for group in all_group_examples_positive]
    augmented_output_negative = [torch.LongTensor(all_label[group]) for group in all_group_examples_negative]


    augmented_aux_flattened_positive = [torch.LongTensor(all_aux_flatten[group]) for group in all_group_examples_positive]
    augmented_aux_flattened_negative = [torch.LongTensor(all_aux_flatten[group]) for group in all_group_examples_negative]


    augmented_aux_positive = [torch.LongTensor(all_aux[group]) for group in all_group_examples_positive]
    augmented_aux_negative = [torch.LongTensor(all_aux[group]) for group in all_group_examples_negative]

    augmented_input = torch.vstack([augmented_input_positive, augmented_input_negative])
    augmented_aux = torch.vstack([augmented_aux_positive[0], augmented_aux_negative[0]])
    augmented_output = torch.hstack([augmented_output_positive[0], augmented_output_negative[0]])
    augmented_aux_flat = torch.hstack([augmented_aux_flattened_positive[0], augmented_aux_flattened_negative[0]])

    shuffle_index = torch.randperm(augmented_input.shape[0])

    batch_input = {
        'augmented_labels': torch.LongTensor(augmented_output)[shuffle_index],
        'input': torch.FloatTensor(augmented_input)[shuffle_index],
        'labels': torch.LongTensor(augmented_output)[shuffle_index],
        'aux': torch.LongTensor(augmented_aux)[shuffle_index],
        'aux_flattened': torch.LongTensor(augmented_aux_flat)[shuffle_index],
        'group_weight_lambda': group_weight_lambda
    }



    return batch_input, np.sum(mask_s)



def simplified_fairness_loss(fairness_function, loss, preds, aux, group1_pattern, group2_pattern, label):
    group1_mask = aux == group1_pattern # where group1 exists
    group2_mask = aux == group2_pattern # where group2 exists

    if fairness_function == 'demographic_parity':
        preds_mask = torch.argmax(preds,1) == 1 # label does not matter here.
        group1_loss = loss[torch.logical_and(preds_mask, group1_mask)]
        group2_loss = loss[torch.logical_and(preds_mask, group2_mask)]
        return torch.abs(torch.mean(group1_loss)-torch.mean(group2_loss))
    elif fairness_function == 'equal_odds' or fairness_function == 'equal_opportunity':
        label_mask_1 = label == 1
        label_mask_0 = label == 0
        final_loss = []


        # true positive rate
        # final_loss = torch.tensor(0.0, requires_grad=True)
        numerator_label = 1
        preds_mask = torch.logical_and(torch.argmax(preds, 1) == numerator_label, label_mask_1)
        group1_loss = loss[torch.logical_and(preds_mask, group1_mask)]
        group2_loss = loss[torch.logical_and(preds_mask, group2_mask)]
        # final_loss.append(torch.abs(torch.mean(group1_loss) - torch.mean(group2_loss)))
        reg_loss = torch.abs(torch.mean(group1_loss) - torch.mean(group2_loss))

        if fairness_function == 'equal_odds':
            # false positive rate
            numerator_label = 0
            preds_mask = torch.logical_and(torch.argmax(preds, 1) == numerator_label, label_mask_0)
            group1_loss = loss[torch.logical_and(preds_mask, group1_mask)]
            group2_loss = loss[torch.logical_and(preds_mask, group2_mask)]
            # final_loss.append(torch.abs(torch.mean(group1_loss) - torch.mean(group2_loss)))
            # reg_loss = torch.max(reg_loss, torch.abs(torch.mean(group1_loss) - torch.mean(group2_loss)))
            reg_loss +=  torch.abs(torch.mean(group1_loss) - torch.mean(group2_loss))

        return reg_loss

    else:
        raise NotImplementedError


def generate_similarity_matrix(iterator, model, groups, reverse_groups, distance_mechanism='dynamic_distance',
                               train_tilted_params=None):
    all_train_label, all_train_s, all_train_s_flatten, all_input = generate_flat_output_custom(
        iterator)

    all_unique_groups = np.unique(all_train_s, axis=0) # unique groups - [[0,0,0], [0,0,1], [0,1,1]]
    all_average_representation = {}


    if distance_mechanism in ["static_distance", "dynamic_distance"]:
        for unique_group in all_unique_groups:
            mask = generate_mask(all_train_s, unique_group)
            current_input = all_input[mask]

            if distance_mechanism == 'static_distance':
                # raise Warning("this has not been tested before. Do test it before hand")
                average_representation = np.mean(current_input, axis=0)
            elif distance_mechanism == 'dynamic_distance':
                batch_input = {
                    'input': torch.FloatTensor(current_input),
                }

                model_hidden = model(batch_input)['hidden']

                average_representation = torch.mean(model_hidden, axis=0).cpu().detach().numpy()
            else:
                raise NotImplementedError

            all_average_representation[tuple([int(i) for i in unique_group])] = average_representation

        # average representation = {str([0,0,1]): average_representation, str([0,1,1]): average_representation}
        distance_lookup = {}

        for unique_group in groups:
            distance = []
            unique_group_representation = all_average_representation[reverse_groups[unique_group]]
            for group in groups:
                distance.append(cosine_distances([unique_group_representation], [all_average_representation[reverse_groups[group]]])[0][0])
            distance_lookup[unique_group] = distance
    elif distance_mechanism == "miixup_distance":
        distance_lookup = {}

        for unique_group in groups:
            distance = []
            for group in groups:
                items_group_0, items_group_1 = sample_data(train_tilted_params, s_group_0=unique_group, s_group_1=group)
                loss_rg = mixup_sub_routine(train_tilted_params, items_group_0, items_group_1, model, gamma=None)
                distance.append(loss_rg.data.item())
            distance_lookup[unique_group] = distance

    return distance_lookup

def training_loop_common(training_loop_parameters: TrainingLoopParameters, train_function):
    logger = logging.getLogger(training_loop_parameters.unique_id_for_run)


    output = {}
    all_train_eps_metrics = []
    all_test_eps_metrics = []
    all_valid_eps_metrics = []
    models = []
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
            fairness_function=training_loop_parameters.fairness_function,
            iterator_set=training_loop_parameters.iterators)

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
            fairness_function=training_loop_parameters.fairness_function,
            iterator_set=None)

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
            fairness_function=training_loop_parameters.fairness_function,
            iterator_set=None)

        test_epoch_metric, loss = train_function(test_parameters)
        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=test_epoch_metric, loss=loss, train=False)
        log_epoch_metric(logger, start_message='test',  epoch_metric=test_epoch_metric, epoch_number=ep, loss=loss)
        all_test_eps_metrics.append(test_epoch_metric)

        logger.info("end of epoch block")

        models.append(copy.deepcopy(training_loop_parameters.model))

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


    index = find_best_model(output, training_loop_parameters.fairness_function)
    output['best_model_index'] = index
    method = training_loop_parameters.other_params['method']
    dataset_name = training_loop_parameters.other_params['dataset_name']
    seed = training_loop_parameters.other_params['seed']
    _dir = f'../saved_models/{dataset_name}/{method}/{seed}'
    Path(_dir).mkdir(parents=True, exist_ok=True)
    if training_loop_parameters.save_model_as != None:
        torch.save(models[index].state_dict(),
                   f'{_dir}/{training_loop_parameters.fairness_function}_{training_loop_parameters.save_model_as}.pt')


    return output