
import math
import torch
import pandas as pd
from scipy import optimize
from tqdm.auto import tqdm
from numpy.random import beta
import torch.nn.functional as F
from collections import Counter
from .common_functionality import *
from metrics import fairness_utils


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

    return epoch_metric_tracker, loss, global_weight, global_loss