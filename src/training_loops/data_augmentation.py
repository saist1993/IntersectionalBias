import math
import torch
import pandas as pd
from analysis import *
from pprint import pprint
from scipy import optimize
from tqdm.auto import tqdm
from numpy.random import beta
import torch.nn.functional as F
from collections import Counter
from metrics import fairness_utils
from .common_functionality import *
from .train_titled_erm_v2 import mixup_sub_routine, sample_data, generate_similarity_matrix

def train_simple_mixup_data_augmentation(train_tilted_params: TrainParameters):
    global_loss = train_tilted_params.other_params['global_loss']  # This tracks the actual loss
    global_weight = train_tilted_params.other_params[
        'global_weight']  # Weights of each examples based on simple count
    # global_weight never gets updated.
    tilt_t = train_tilted_params.other_params['titled_t']  # This should be small. In order of magnitude of 0.01

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []
    group_tracker = [0 for _ in range(len(train_tilted_params.other_params['groups']))]



    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        alpha = 1.0
        gamma = beta(alpha, alpha)
        s1, s2 = np.random.choice(train_tilted_params.other_params['groups'], 2, p=global_weight)

        # items_s1 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
        #                              train_tilted_params.other_params['all_label'],
        #                              train_tilted_params.other_params['all_aux'],
        #                              train_tilted_params.other_params['all_aux_flatten'],
        #                              train_tilted_params.other_params['batch_size'],
        #                              s1)
        #
        #
        # items_s2 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
        #                              train_tilted_params.other_params['all_label'],
        #                              train_tilted_params.other_params['all_aux'],
        #                              train_tilted_params.other_params['all_aux_flatten'],
        #                              train_tilted_params.other_params['batch_size'],
        #                              s2)

        # do the mixup

        items_s1, items_s2 = sample_data(train_tilted_params, s1, s2)

        items = {}

        items['input'] = items_s1['input'] * gamma + items_s2['input'] * (1 - gamma)
        # items['labels'] = items_s1['labels']*gamma + items_s2['labels']*(1-gamma)
        items['labels'] = torch.nn.functional.one_hot(items_s1['labels'])*gamma + torch.nn.functional.one_hot(items_s2['labels'])*(1-gamma)
        items['aux'] = items_s1['aux']  # this is a hack
        items['aux_flattened'] = items_s1['aux_flattened']

        for key in items.keys():
            try:
                items[key] = items[key].to(train_tilted_params.device)
            except AttributeError:
                continue

        optimizer.zero_grad()
        output = model(items)
        loss = torch.mean(criterion(output['prediction'], items['labels'])) \
               + mixup_sub_routine(train_tilted_params, items_s1, items_s2, model, gamma)
        loss.backward()
        optimizer.step()


        # tracking the original input instead of augmented one.
        with torch.no_grad():
            output = model(items_s1)
            track_output.append(output)
            track_input.append(items_s1)
        output['loss_batch'] = loss.item()

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    return epoch_metric_tracker, loss, global_weight, global_loss


def train_lisa_based_mixup(train_tilted_params: TrainParameters):
    """
    possible methods - lisa_based_mixup; lisa_based_mixup_with_mixup_regularizer, lisa_based_mixup_with_distance, lisa_based_mixup_with_mixup_regularizer_and_with_distance
    :param train_tilted_params:
    :return:
    """
    global_loss = train_tilted_params.other_params['global_loss']  # This tracks the actual loss
    global_weight = train_tilted_params.other_params[
        'global_weight']  # Weights of each examples based on simple count
    # global_weight never gets updated.
    tilt_t = train_tilted_params.other_params['titled_t']  # This should be small. In order of magnitude of 0.01
    method = train_tilted_params.other_params['method']

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []
    group_tracker = [0 for _ in range(len(train_tilted_params.other_params['groups']))]

    mixup_regularizer = False
    based_on_distance = False

    if "with_distance" in method:
        based_on_distance = True
    if "mixup_regularizer" in method:
        mixup_regularizer = True

    print(based_on_distance, mixup_regularizer)

    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}
    similarity_matrix = generate_similarity_matrix(train_tilted_params.other_params['valid_iterator'], model,
                                                   train_tilted_params.other_params['groups'], flattened_s_to_s)


    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        alpha = 1.0
        gamma = beta(alpha, alpha)
        if based_on_distance:
            s1 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
            s_group_distance = similarity_matrix[s1]
            s2 = np.random.choice(train_tilted_params.other_params['groups'], 1, replace=False,
                                         p=s_group_distance / np.linalg.norm(s_group_distance, 1))[0]
        else:
            s1, s2 = np.random.choice(train_tilted_params.other_params['groups'], 2, p=global_weight)


        ps = np.random.uniform(0, 1)

        optimizer.zero_grad() # There is a reason this is here.


        if ps > 0.5:
            items_s1 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                        train_tilted_params.other_params['all_label'],
                                                        train_tilted_params.other_params['all_aux'],
                                                        train_tilted_params.other_params['all_aux_flatten'],
                                                        train_tilted_params.other_params['batch_size'],
                                                        s1)


            items_s2 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                        train_tilted_params.other_params['all_label'],
                                                        train_tilted_params.other_params['all_aux'],
                                                        train_tilted_params.other_params['all_aux_flatten'],
                                                        train_tilted_params.other_params['batch_size'],
                                                        s1) # this is not a bug

            if mixup_regularizer:
                regulizer = mixup_sub_routine(train_tilted_params, items_s1, items_s2, model, gamma)
            split_index = int(train_tilted_params.other_params['batch_size'] / 2)

            items_s1['input'] = torch.vstack([items_s1['input'][split_index:], items_s1['input'][:split_index]])
            items_s1['aux'] = torch.vstack([items_s1['aux'][split_index:], items_s1['aux'][:split_index]])
            items_s1['labels'] = torch.hstack([items_s1['labels'][split_index:], items_s1['labels'][:split_index]])
            items_s1['aux_flattened'] = torch.hstack([items_s1['aux_flattened'][split_index:], items_s1['aux_flattened'][:split_index]])


        else:
            items_s1 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                   train_tilted_params.other_params['all_label'],
                                                   train_tilted_params.other_params['all_aux'],
                                                   train_tilted_params.other_params['all_aux_flatten'],
                                                   train_tilted_params.other_params['batch_size'],
                                                   s1)

            items_s2 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                   train_tilted_params.other_params['all_label'],
                                                   train_tilted_params.other_params['all_aux'],
                                                   train_tilted_params.other_params['all_aux_flatten'],
                                                   train_tilted_params.other_params['batch_size'],
                                                   s2)

            regulizer = mixup_sub_routine(train_tilted_params, items_s1, items_s2, model, gamma)

        items = {}

        items['input'] = items_s1['input'] * gamma + items_s2['input'] * (1 - gamma)
        # items['labels'] = items_s1['labels']*gamma + items_s2['labels']*(1-gamma)
        items['labels'] = torch.nn.functional.one_hot(items_s1['labels'])*gamma + torch.nn.functional.one_hot(items_s2['labels'])*(1-gamma)
        items['aux'] = items_s1['aux']  # this is a hack
        items['aux_flattened'] = items_s1['aux_flattened']

        for key in items.keys():
            try:
                items[key] = items[key].to(train_tilted_params.device)
            except AttributeError:
                continue


        output = model(items)
        loss = torch.mean(criterion(output['prediction'], items['labels']))
        if mixup_regularizer:
            loss = loss + regulizer

        loss.backward()
        optimizer.step()


        # tracking the original input instead of augmented one.
        with torch.no_grad():
            output = model(items_s1)
            track_output.append(output)
            track_input.append(items_s1)
        output['loss_batch'] = loss.item()

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    return epoch_metric_tracker, loss, global_weight, global_loss


def generate_combinations_take_2(s, inverse_dict):
    def flip_bit(bit):
        if bit == 1:
            return 0
        else:
            return 1

    all_s_combinations = []

    for index, i in enumerate(s):
        temp = list(copy.deepcopy(s))
        temp[index] = flip_bit(i)
        all_s_combinations.append(inverse_dict[tuple(temp)])

    return all_s_combinations




def generate_combinations_take_3(s, inverse_dict):
    def flip_bit(bit):
        if bit == 1:
            return 0
        else:
            return 1

    all_s_combinations = []

    for index1, index2 in combinations(range(len(s)), 2):
        temp = list(copy.deepcopy(s))
        temp[index1] = flip_bit(s[index1])
        temp[index2] = flip_bit(s[index2])
        all_s_combinations.append(inverse_dict[tuple(temp)])

    return all_s_combinations

def train_lisa_based_mixup_take_2(train_tilted_params: TrainParameters):
    """
    possible methods - lisa_based_mixup; lisa_based_mixup_with_mixup_regularizer, lisa_based_mixup_with_distance, lisa_based_mixup_with_mixup_regularizer_and_with_distance
    :param train_tilted_params:
    :return:
    """
    global_loss = train_tilted_params.other_params['global_loss']  # This tracks the actual loss
    global_weight = train_tilted_params.other_params[
        'global_weight']  # Weights of each examples based on simple count
    # global_weight never gets updated.
    tilt_t = train_tilted_params.other_params['titled_t']  # This should be small. In order of magnitude of 0.01
    method = train_tilted_params.other_params['method']

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []
    group_tracker = [0 for _ in range(len(train_tilted_params.other_params['groups']))]

    mixup_regularizer = False
    based_on_distance = False

    # if "with_distance" in method:
    #     based_on_distance = True
    if "mixup_regularizer" in method:
        mixup_regularizer = True

    print(based_on_distance, mixup_regularizer)

    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}
    similarity_matrix = generate_similarity_matrix(train_tilted_params.other_params['valid_iterator'], model,
                                                   train_tilted_params.other_params['groups'], flattened_s_to_s)


    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        alpha = 1.0
        gamma = beta(alpha, alpha)



        ps = np.random.uniform(0, 1)

        optimizer.zero_grad() # There is a reason this is here.

        s1 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]

        if ps > 0.5:

            s2 = np.random.choice(generate_combinations_take_2(flattened_s_to_s[s1], train_tilted_params.other_params['s_to_flattened_s']),1)[0]


            items_s1 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                        train_tilted_params.other_params['all_label'],
                                                        train_tilted_params.other_params['all_aux'],
                                                        train_tilted_params.other_params['all_aux_flatten'],
                                                        train_tilted_params.other_params['batch_size'],
                                                        s1)


            items_s2 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                        train_tilted_params.other_params['all_label'],
                                                        train_tilted_params.other_params['all_aux'],
                                                        train_tilted_params.other_params['all_aux_flatten'],
                                                        train_tilted_params.other_params['batch_size'],
                                                        s2) # this is not a bug

            if mixup_regularizer:
                regulizer = mixup_sub_routine(train_tilted_params, items_s1, items_s2, model, gamma)
            split_index = int(train_tilted_params.other_params['batch_size'] / 2)

            items_s1['input'] = torch.vstack([items_s1['input'][split_index:], items_s1['input'][:split_index]])
            items_s1['aux'] = torch.vstack([items_s1['aux'][split_index:], items_s1['aux'][:split_index]])
            items_s1['labels'] = torch.hstack([items_s1['labels'][split_index:], items_s1['labels'][:split_index]])
            items_s1['aux_flattened'] = torch.hstack([items_s1['aux_flattened'][split_index:], items_s1['aux_flattened'][:split_index]])


        else:
            s2 = np.random.choice(generate_combinations_take_3(flattened_s_to_s[s1], train_tilted_params.other_params['s_to_flattened_s']), 1)[0]
            items_s1 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                   train_tilted_params.other_params['all_label'],
                                                   train_tilted_params.other_params['all_aux'],
                                                   train_tilted_params.other_params['all_aux_flatten'],
                                                   train_tilted_params.other_params['batch_size'],
                                                   s1)

            items_s2 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                   train_tilted_params.other_params['all_label'],
                                                   train_tilted_params.other_params['all_aux'],
                                                   train_tilted_params.other_params['all_aux_flatten'],
                                                   train_tilted_params.other_params['batch_size'],
                                                   s2)

            regulizer = mixup_sub_routine(train_tilted_params, items_s1, items_s2, model, gamma)

        items = {}

        items['input'] = items_s1['input'] * gamma + items_s2['input'] * (1 - gamma)
        # items['labels'] = items_s1['labels']*gamma + items_s2['labels']*(1-gamma)
        items['labels'] = torch.nn.functional.one_hot(items_s1['labels'])*gamma + torch.nn.functional.one_hot(items_s2['labels'])*(1-gamma)
        items['aux'] = items_s1['aux']  # this is a hack
        items['aux_flattened'] = items_s1['aux_flattened']

        for key in items.keys():
            try:
                items[key] = items[key].to(train_tilted_params.device)
            except AttributeError:
                continue


        output = model(items)
        loss = torch.mean(criterion(output['prediction'], items['labels']))
        if mixup_regularizer:
            loss = loss + regulizer

        loss.backward()
        optimizer.step()


        # tracking the original input instead of augmented one.
        with torch.no_grad():
            output = model(items_s1)
            track_output.append(output)
            track_input.append(items_s1)
        output['loss_batch'] = loss.item()

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    return epoch_metric_tracker, loss, global_weight, global_loss

def train_only_group_dro_with_mixup_regularizer_super_group_data_augmentation(train_tilted_params:TrainParameters):
    global_loss = train_tilted_params.other_params['global_loss'] # This tracks the actual loss
    global_weight = train_tilted_params.other_params['global_weight'] # Weights of each examples based on simple count
    # global_weight never gets updated.
    tilt_t = train_tilted_params.other_params['titled_t'] # This should be small. In order of magnitude of 0.01
    method = train_tilted_params.other_params['method'] # This should be small. In order of magnitude of 0.01

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []
    group_tracker = [0 for _ in range(len(train_tilted_params.other_params['groups']))]
    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}
    similarity_matrix = generate_similarity_matrix(train_tilted_params.other_params['valid_iterator'], model,
                                                   train_tilted_params.other_params['groups'], flattened_s_to_s)

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        alpha = 1.0
        gamma = beta(alpha, alpha)

        s_group_0, s_group_1 = eval(
            np.random.choice(train_tilted_params.other_params['groups_matrix'].reshape(1, -1)[0], 1, replace=False,
                             p=global_weight.reshape(1, -1)[0])[0])

        items_s1, items_s2 = sample_data(train_tilted_params, s_group_0, s_group_1)

        group_tracker[s_group_0] += 1
        group_tracker[s_group_1] += 1

        optimizer.zero_grad()

        items = {}

        items['input'] = items_s1['input'] * gamma + items_s2['input'] * (1 - gamma)
        # items['labels'] = items_s1['labels']*gamma + items_s2['labels']*(1-gamma)
        items['labels'] = torch.nn.functional.one_hot(items_s1['labels'])*gamma + torch.nn.functional.one_hot(items_s2['labels'])*(1-gamma)
        items['aux'] = items_s1['aux']  # this is a hack
        items['aux_flattened'] = items_s1['aux_flattened']

        for key in items.keys():
            try:
                items[key] = items[key].to(train_tilted_params.device)
            except AttributeError:
                continue
        output = model(items)
        loss = torch.mean(criterion(output['prediction'], items['labels']))

        if 'mixup_regularizer' in method:
            loss_reg = mixup_sub_routine(train_tilted_params, items_s1, items_s2, model, gamma)
            loss = loss + loss_reg


        global_loss[s_group_0, s_group_1] = global_loss[s_group_0, s_group_1] * torch.exp(tilt_t*loss.data)
        global_loss = global_loss/(global_loss.sum())
        loss = global_loss[s_group_0, s_group_1]*loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            output = model(items_s1)
            track_output.append(output)
            track_input.append(items_s1)
        output['loss_batch'] = loss.item()

        track_output.append(output)
        track_input.append(items_s1)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    print(global_loss)
    print(global_weight)
    print(group_tracker)

    return epoch_metric_tracker, loss, global_weight, global_loss