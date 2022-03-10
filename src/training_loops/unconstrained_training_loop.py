import torch
import numpy as np
import torch.nn as nn
from utils import misc
from tqdm.auto import tqdm
from dataclasses import dataclass
from utils import fairness_function as fairness_functions
from typing import NamedTuple, List, Dict, Callable

@dataclass
class TrainParameters:
    model: nn.Module
    iterator: Dict
    optimizer: torch.optim
    criterion: Callable
    device: torch.device
    other_params: Dict
    per_epoch_metric: Callable
    mode: str


@dataclass
class TrainingLoopParameters:
    n_epochs:int
    model: nn.Module
    iterators: Dict
    optimizer: torch.optim
    criterion: Callable
    device: torch.device
    save_model_as: str
    other_params: Dict


@dataclass
class FairnessMetricAnalytics:
    maximum: float
    minimum: float
    average: float
    standard_dev: float


@dataclass
class FairnessMetricTracker:
    gerrymandering: FairnessMetricAnalytics
    independent: FairnessMetricAnalytics
    intersectional: FairnessMetricAnalytics


@dataclass
class EpochMetricTracker():
    accuracy: float
    accuracy_parity: FairnessMetricTracker


def generate_fairness_metric_analytics(results:list):
    fairness_metric_analytics = FairnessMetricAnalytics(
        maximum=max(results),
        minimum=min(results),
        average=np.mean(results),
        standard_dev=np.std(results)
    )
    return fairness_metric_analytics

def accuracy_parity_fairness_metric(prediction, label,all_possible_groups, all_possible_groups_mask):
    accuracy_over_all_groups = np.asarray(fairness_functions.accuracy_parity_over_groups(prediction, label,
                                                                              all_possible_groups_mask))

    # gerrymandering
    _, gerrymandering_index = fairness_functions.get_gerrymandering_groups(all_possible_groups)
    gerrymandering_accuracy = accuracy_over_all_groups[gerrymandering_index]
    gerrymandering_fairness_metric_analytics = generate_fairness_metric_analytics(gerrymandering_accuracy)

    # independent
    _, independent_index = fairness_functions.get_independent_groups(all_possible_groups)
    independent_accuracy = accuracy_over_all_groups[independent_index]
    independent_fairness_metric_analytics = generate_fairness_metric_analytics(independent_accuracy)

    # intersectional
    _, intersectional_index = fairness_functions.get_intersectional_groups(all_possible_groups)
    intersectional_accuracy = accuracy_over_all_groups[intersectional_index]
    intersectional_fairness_metric_analytics = generate_fairness_metric_analytics(intersectional_accuracy)

    fairness_metric_tracker = FairnessMetricTracker(gerrymandering=gerrymandering_fairness_metric_analytics,
                          independent=independent_fairness_metric_analytics,
                          intersectional=intersectional_fairness_metric_analytics)

    return fairness_metric_tracker



def calculate_fairness(prediction, label, aux):
    """This is going to call the whole fairness routines"""

    # Create masks for every possible group
    all_possible_groups = fairness_functions.create_all_possible_groups(number_of_attributes=aux.shape[1])
    all_possible_groups_mask = [fairness_functions.create_mask(data=aux, condition=group) for group in all_possible_groups]

    # Calculate rates for every possible group. This includes: accuracy, true-positive-rate, and other possible rates
    accuracy_parity_fairness_metric_tracker = accuracy_parity_fairness_metric(prediction, label,all_possible_groups,all_possible_groups_mask)

    return accuracy_parity_fairness_metric_tracker


def per_epoch_metric(epoch_output, epoch_input):
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
    for batch_output, batch_input in zip(epoch_output, epoch_input):
        all_prediction.append(batch_output['prediction'].detach().numpy())
        all_loss.append(batch_output['loss_batch'])
        all_label.append(batch_input['labels'].numpy())
        all_s.append(batch_input['aux'].numpy())
    all_prediction = np.vstack(all_prediction).argmax(1)
    all_label = np.hstack(all_label)
    all_s = np.vstack(all_s)

    # Calculate accuracy
    accuracy = fairness_functions.calculate_accuracy_classification(predictions=all_prediction, labels=all_label)

    # Calculate fairness
    accuracy_parity_fairness_metric_tracker = calculate_fairness(prediction=all_prediction, label=all_label, aux=all_s)
    epoch_metric = EpochMetricTracker(accuracy=accuracy, accuracy_parity=accuracy_parity_fairness_metric_tracker)
    return accuracy, epoch_metric


def train(train_parameters: TrainParameters):
    """Trains the model for one epoch"""
    model, optimizer, device, criterion, mode =\
        train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion, train_parameters.mode

    if mode == 'train':
        model.train()
    elif mode == 'evaluate':
        model.eval()
    else:
        raise misc.CustomError("only supports train and evaluate")
    track_output = []

    for items in tqdm(train_parameters.iterator):

        # Change the device of all the tensors!
        for key in items.keys():
            items[key] = items[key].to(device)

        if mode == 'train':
            optimizer.zero_grad()
            output = model(items)
            loss = torch.mean(criterion(output['prediction'], items['labels'])) # As reduction is None.
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

    # Calculate all per-epoch metric
    epoch_metric_tracker = train_parameters.per_epoch_metric(track_output,train_parameters.iterator)
    return epoch_metric_tracker


def training_loop(training_loop_parameters: TrainingLoopParameters):

    for epoch in range(training_loop_parameters.n_epochs):

        train_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['train_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='train')
        train_accuracy = train(train_parameters)



        test_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['test_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='evaluate')

        test_accuracy = train(test_parameters)
        print(f"train accuracy is {train_accuracy}")
        print(f"test accuracy is {test_accuracy}")



