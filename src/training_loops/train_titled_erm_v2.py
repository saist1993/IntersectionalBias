
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
from .mixup_training_loop import generate_similarity_matrix
from .titled_erm_with_abstract import sample_batch_sen_idx_with_augmentation_with_lambda_custom_with_positive_and_negative_seperate

def train_only_tilted_erm_generic(train_tilted_params:TrainParameters):
    """
    Possible methods
        - only_tilted_erm_generic
        - only_tilted_erm_with_mask_on_tpr
        - only_tilted_erm_with_weighted_loss_via_global_weight
        - only_tilted_erm_with_mask_on_tpr_and_weighted_loss_via_global_weight
    """

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

    method = train_tilted_params.other_params['method']

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]

        items = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s)

        for key in items.keys():
            items[key] = items[key].to(train_tilted_params.device)

        optimizer.zero_grad()
        output = model(items)
        loss = criterion(output['prediction'], items['labels'])

        if 'with_mask_on_tpr' in method:
            if train_tilted_params.fairness_function == 'equal_opportunity':
                mask = items['labels'] == 1
                loss_without_backward = torch.mean(torch.clone(loss[mask])).detach()
            else:
                loss_without_backward = torch.mean(torch.clone(loss)).detach()
        else:
            loss_without_backward = torch.mean(torch.clone(loss)).detach()
        loss = torch.mean(loss)


        global_weight[s] =  0.2 * torch.exp(tilt_t*loss_without_backward) + 0.8 * global_weight[s]
        global_weight = global_weight / torch.sum(global_weight)

        # global_loss[s] =  0.2 * torch.exp(tilt_t*loss_without_backward) + 0.8 * global_loss[s]

        # weights = torch.exp(tao*loss_without_backward - tao*global_loss[s])
        # global_weight = global_loss / torch.sum(global_loss)

        if 'weighted_loss_via_global_weight' in method:
            loss = global_weight[s] * loss

        loss.backward()
        optimizer.step()


        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)




    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)

    print(global_weight)


    return epoch_metric_tracker, loss, global_weight, global_loss


def train_only_group_dro(train_tilted_params:TrainParameters):
    global_loss = train_tilted_params.other_params['global_loss'] # This tracks the actual loss
    global_weight = train_tilted_params.other_params['global_weight'] # Weights of each examples based on simple count
    # global_weight never gets updated.
    tilt_t = train_tilted_params.other_params['titled_t'] # This should be small. In order of magnitude of 0.01

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []
    group_tracker = [0 for _ in range(len(train_tilted_params.other_params['groups']))]


    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]

        items = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s)

        group_tracker[s] += 1


        for key in items.keys():
            items[key] = items[key].to(train_tilted_params.device)

        optimizer.zero_grad()
        output = model(items)
        loss = torch.mean(criterion(output['prediction'], items['labels']))
        # loss = loss + 2/np.sqrt(train_tilted_params.other_params['group_count'][s])
        global_loss[s] = global_loss[s] * torch.exp(tilt_t*loss.data)
        global_loss = global_loss/(global_loss.sum())
        loss = loss*global_loss[s]

        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    print(global_loss)
    print(global_weight)
    print(group_tracker)
    # pprint(run_equal_odds(model=model,iterators=train_tilted_params.iterator,criterion=criterion,mode=None))

    return epoch_metric_tracker, loss, global_weight, global_loss


def train_only_group_dro_with_augmentation_static_positive_and_negative_weights(train_tilted_params:TrainParameters):
    global_loss = train_tilted_params.other_params['global_loss'] # This tracks the actual loss
    global_weight = train_tilted_params.other_params['global_weight'] # Weights of each examples based on simple count
    # global_weight never gets updated.
    tilt_t = train_tilted_params.other_params['titled_t'] # This should be small. In order of magnitude of 0.01

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []
    group_tracker = [0 for _ in range(len(train_tilted_params.other_params['groups']))]
    flattened_s_to_s = {value: key for key, value in train_tilted_params.other_params['s_to_flattened_s'].items()}
    group_to_lambda_weight = train_tilted_params.other_params['group_to_lambda_weight']

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        s_flat = flattened_s_to_s[s]


        _items = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s)

        items, size_of_s = sample_batch_sen_idx_with_augmentation_with_lambda_custom_with_positive_and_negative_seperate(
            train_tilted_params.other_params['all_input'],
            train_tilted_params.other_params['all_label'],
            train_tilted_params.other_params['all_aux'],
            train_tilted_params.other_params['all_aux_flatten'],
            train_tilted_params.other_params['batch_size'],
            s_flat,
            group_to_lambda_weight[s],
            train_tilted_params.other_params['scalar'],
            _items)

        group_tracker[s] += 1


        for key in items.keys():
            try:
                items[key] = items[key].to(train_tilted_params.device)
            except AttributeError:
                continue

        optimizer.zero_grad()
        output = model(items)
        loss = torch.mean(criterion(output['prediction'], items['labels']))
        # loss = loss + 2/np.sqrt(train_tilted_params.other_params['group_count'][s])
        global_loss[s] = global_loss[s] * torch.exp(tilt_t*loss.data)
        global_loss = global_loss/(global_loss.sum())
        loss = loss*global_loss[s]

        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    print(global_loss)
    print(global_weight)
    print(group_tracker)
    # pprint(run_equal_odds(model=model,iterators=train_tilted_params.iterator,criterion=criterion,mode=None))

    return epoch_metric_tracker, loss, global_weight, global_loss


def mixup_sub_routine(train_tilted_params:TrainParameters, items_group_0, items_group_1, model, gamma=None):
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
                                  1 - gamma)
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


def sample_data(train_tilted_params, s_group_0, s_group_1):
    if train_tilted_params.fairness_function == 'demographic_parity':
        items_group_0 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                             train_tilted_params.other_params['all_label'],
                                             train_tilted_params.other_params['all_aux'],
                                             train_tilted_params.other_params['all_aux_flatten'],
                                             train_tilted_params.other_params['batch_size'],
                                             s_group_0)

        items_group_1 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                             train_tilted_params.other_params['all_label'],
                                             train_tilted_params.other_params['all_aux'],
                                             train_tilted_params.other_params['all_aux_flatten'],
                                             train_tilted_params.other_params['batch_size'],
                                             s_group_1)

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
        items_group_1 = sample_batch_sen_idx_with_y(train_tilted_params.other_params['all_input'],
                                                                   train_tilted_params.other_params['all_label'],
                                                                   train_tilted_params.other_params['all_aux'],
                                                                   train_tilted_params.other_params['all_aux_flatten'],
                                                                   train_tilted_params.other_params['batch_size'],
                                                                   s_group_1)
        # group split

        # class split

    else:
        raise NotImplementedError

    for key in items_group_0.keys():
        items_group_0[key] = items_group_0[key].to(train_tilted_params.device)

    for key in items_group_1.keys():
        items_group_1[key] = items_group_1[key].to(train_tilted_params.device)

    return items_group_0, items_group_1



def train_only_group_dro_with_mixup(train_tilted_params:TrainParameters):
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
        if 'mixup_with_distance' in method:
            s_group_0 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
            s_group_distance = similarity_matrix[s_group_0]
            s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 1, replace=False,
                                         p=s_group_distance / np.linalg.norm(s_group_distance, 1))[0]
        elif 'mixup_with_random' in method:
            s_group_0, s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 2, replace=False, p=global_weight)
            # Is this the most optimum way. here if global weights are same then it is not an issue. But if
            # global weight represents actual count, then we will sample two groups with least count often.
        else:
            raise NotImplementedError

        items_group_0, items_group_1 = sample_data(train_tilted_params, s_group_0, s_group_1)

        group_tracker[s_group_0] += 1
        group_tracker[s_group_1] += 1

        optimizer.zero_grad()

        output_group_0 = model(items_group_0)
        output_group_1 = model(items_group_1)

        loss_reg = mixup_sub_routine(train_tilted_params, items_group_0, items_group_1, model)

        loss_group_0 = torch.mean(criterion(output_group_0['prediction'], items_group_0['labels'])) + loss_reg
        loss_group_1 = torch.mean(criterion(output_group_1['prediction'], items_group_1['labels'])) + loss_reg


        global_loss[s_group_0] = global_loss[s_group_0] * torch.exp(tilt_t*loss_group_0.data)
        global_loss[s_group_1] = global_loss[s_group_1] * torch.exp(tilt_t*loss_group_1.data)

        # global_loss[s_group_0] = global_loss[s_group_0] * torch.exp(tilt_t * (loss_group_0.data + loss_group_1.data))
        # global_loss[s_group_1] = global_loss[s_group_1] * torch.exp(tilt_t * (loss_group_1.data + loss_group_1.data))


        global_loss = global_loss/(global_loss.sum())
        loss = loss_group_0*global_loss[s_group_0] + loss_group_1*global_loss[s_group_1]
        loss.backward()
        optimizer.step()

        output_group_0['loss_batch'] = loss_group_0.item()
        track_output.append(output_group_0)
        track_input.append(items_group_0)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    print(global_loss)
    print(global_weight)
    print(group_tracker)

    return epoch_metric_tracker, loss, global_weight, global_loss



def train_only_group_dro_with_mixup_regularizer_super_group(train_tilted_params:TrainParameters):
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
        s_group_0, s_group_1 = eval(
            np.random.choice(train_tilted_params.other_params['groups_matrix'].reshape(1, -1)[0], 1, replace=False,
                             p=global_weight.reshape(1, -1)[0])[0])

        items_group_0, items_group_1 = sample_data(train_tilted_params, s_group_0, s_group_1)

        group_tracker[s_group_0] += 1
        group_tracker[s_group_1] += 1

        optimizer.zero_grad()

        output_group_0 = model(items_group_0)
        output_group_1 = model(items_group_1)



        loss_group_0 = torch.mean(criterion(output_group_0['prediction'], items_group_0['labels']))
        loss_group_1 = torch.mean(criterion(output_group_1['prediction'], items_group_1['labels']))
        loss = loss_group_0 + loss_group_1

        if 'mixup_regularizer' in method:
            loss_reg = mixup_sub_routine(train_tilted_params, items_group_0, items_group_1, model)
            loss = loss + loss_reg


        global_loss[s_group_0, s_group_1] = global_loss[s_group_0, s_group_1] * torch.exp(tilt_t*loss.data)
        global_loss = global_loss/(global_loss.sum())
        loss = global_loss[s_group_0, s_group_1]*loss
        loss.backward()
        optimizer.step()

        output_group_0['loss_batch'] = loss_group_0.item()
        track_output.append(output_group_0)
        track_input.append(items_group_0)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                      track_input,
                                                                      train_tilted_params.fairness_function)

    print(global_loss)
    print(global_weight)
    print(group_tracker)

    return epoch_metric_tracker, loss, global_weight, global_loss
