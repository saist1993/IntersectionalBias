from dataclasses import dataclass
from typing import Dict, Callable, Optional

import torch
import wandb
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from itertools import combinations


from metrics import calculate_epoch_metric, fairness_utils
from .common_functionality import *
from utils import misc


def train(train_parameters: TrainParameters):
    """Trains the model for one epoch"""
    model, optimizer, device, criterion, mode = \
        train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion, train_parameters.mode

    if mode == 'train':
        model.train()
    elif mode == 'evaluate':
        model.eval()
    else:
        raise misc.CustomError("only supports train and evaluate")
    track_output = []

    all_s = []
    for items in train_parameters.iterator:
        all_s.append(items['aux'].numpy())
    all_s = np.vstack(all_s)
    all_independent_group_patterns = fairness_utils.create_all_possible_groups \
        (attributes=[list(np.unique(all_s[:, i])) for i in range(all_s.shape[1])])
    all_independent_group_patterns = [torch.tensor(i) for i in all_independent_group_patterns if 'x' not in i]



    for items in tqdm(train_parameters.iterator):

        # Change the device of all the tensors!
        for key in items.keys():
            items[key] = items[key].to(device)

        if mode == 'train':
            optimizer.zero_grad()
            output = model(items)

            loss = criterion(output['prediction'], items['labels'], items['aux_flattened'], mode='train')

            if train_parameters.other_params['fairness_lambda'] != 0.0:
                fairness_loss = \
                    get_fairness_loss \
                        (train_parameters.fairness_function, loss, output['prediction'], items['aux'], items['labels'],
                         all_independent_group_patterns)
                if fairness_loss:
                    loss = torch.mean(loss) + train_parameters.other_params['fairness_lambda'] * fairness_loss
                else:
                    loss = torch.mean(loss)
            else:
                loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
        elif mode == 'evaluate':
            with torch.no_grad():
                output = model(items)
                loss = torch.mean(criterion(output['prediction'], items['labels'], mode='eval'))  # As reduction is None.
        else:
            raise misc.CustomError("only supports train and evaluate")

        # Save all batch stuff for further analysis
        output['loss_batch'] = loss.item()
        track_output.append(output)

    # Calculate all per-epoch metric by sending outputs and the inputs
    epoch_metric_tracker, loss = train_parameters.per_epoch_metric(track_output,
                                                                   train_parameters.iterator,
                                                                   train_parameters.fairness_function)
    return epoch_metric_tracker, loss


def training_loop(training_loop_parameters: TrainingLoopParameters):
    output = training_loop_common(training_loop_parameters, train)
    return output
