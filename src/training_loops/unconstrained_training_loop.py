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


# def per_epoch_metric(epoch_output, epoch_input):
#     """
#     :param epoch_output: access all the batch output.
#     :param epoch_input: Iterator to access the gold data and inputs
#     :return:
#     """
#     # Step1: Flatten everything for easier analysis
#     all_label = []
#     all_prediction = []
#     all_loss = []
#     all_s = []
#     for batch_output, batch_input in zip(epoch_output, epoch_input):
#         all_prediction.append(batch_output['prediction'].detach().numpy())
#         all_loss.append(batch_output['loss_batch'])
#         all_label.append(batch_input['labels'].numpy())
#         all_s.append(batch_input['aux'].numpy())
#     all_prediction = np.vstack(all_prediction).argmax(1)
#     all_label = np.hstack(all_label)
#     all_s = np.vstack(all_s)
#
#     # Calculate accuracy
#     # accuracy = fairness_functions.calculate_accuracy_classification(predictions=all_prediction, labels=all_label)
#     #
#     # # Calculate fairness
#     # accuracy_parity_fairness_metric_tracker, true_positive_rate_fairness_metric_tracker\
#     #     = calculate_fairness(prediction=all_prediction, label=all_label, aux=all_s)
#     # epoch_metric = EpochMetricTracker(accuracy=accuracy, accuracy_parity=accuracy_parity_fairness_metric_tracker,
#     #                                   true_positive_rate=true_positive_rate_fairness_metric_tracker)
#
#     other_meta_data = {
#         # 'fairness_mode': ['demographic_parity', 'equal_opportunity', 'equal_odds'],
#         'fairness_mode': ['equal_opportunity'],
#         'no_fairness': False,
#         'adversarial_output': None
#     }
#
#     epoch_metric = calculate_epoch_metric.CalculateEpochMetric(all_prediction, all_label, all_s, other_meta_data).run()
#     return epoch_metric, np.mean(all_loss)


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
            loss = criterion(output['prediction'], items['labels'])
            fairness_loss = \
                get_fairness_loss_equal_opportunity \
                    (loss, output['prediction'], items['aux'], items['labels'], all_independent_group_patterns)
            if fairness_loss:
                loss = torch.mean(loss) + 0.05*fairness_loss
            else:
                loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
        elif mode == 'evaluate':
            with torch.no_grad():
                output = model(items)
                loss = torch.mean(criterion(output['prediction'], items['labels']))  # As reduction is None.
        else:
            raise misc.CustomError("only supports train and evaluate")

        # Save all batch stuff for further analysis
        output['loss_batch'] = loss.item()
        track_output.append(output)

    # Calculate all per-epoch metric by sending outputs and the inputs
    epoch_metric_tracker, loss = train_parameters.per_epoch_metric(track_output, train_parameters.iterator)
    return epoch_metric_tracker, loss


def training_loop(training_loop_parameters: TrainingLoopParameters):
    output = training_loop_common(training_loop_parameters, train)
    return output
