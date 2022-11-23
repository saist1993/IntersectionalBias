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
from .train_titled_erm_v2 import mixup_sub_routine

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
    alpha = 1.0
    gamma = beta(alpha, alpha)


    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s1, s2 = np.random.choice(train_tilted_params.other_params['groups'], 2, p=global_weight)

        items_s1 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s1)


        items_s2 = sample_batch_sen_idx(train_tilted_params.other_params['all_input'],
                                     train_tilted_params.other_params['all_label'],
                                     train_tilted_params.other_params['all_aux'],
                                     train_tilted_params.other_params['all_aux_flatten'],
                                     train_tilted_params.other_params['batch_size'],
                                     s1)

        # do the mixup

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
        loss = torch.mean(criterion(output['prediction'], items['labels']))
               # + mixup_sub_routine(train_tilted_params, items_s1, items_s2, model)
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