import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from dataclasses import dataclass
from utils import misc
from typing import NamedTuple, List, Dict, Callable

@dataclass
class TrainParameters():
    model: nn.Module
    iterator: Dict
    optimizer: torch.optim
    criterion: Callable
    device: torch.device
    other_params: Dict
    per_epoch_metric: Callable
    mode: str

@dataclass
class TrainingLoopParameters():
    n_epochs:int
    model: nn.Module
    iterators: Dict
    optimizer: torch.optim
    criterion: Callable
    device: torch.device
    save_model_as: str
    other_params: Dict


def calculate_accuracy_classification(predictions, labels):
    top_predictions = predictions.argmax(1)
    correct = (top_predictions == labels).sum()
    accuracy = correct*1.0/ labels.shape[0]
    return accuracy


def per_epoch_metric(epoch_output, epoch_input):
    """
    :param epoch_output: access all the batch output.
    :param epoch_input: Iterator to access the gold data and inputs
    :return:
    """
    # Step1: calculate accuracy
    all_labels = []
    all_prediction = []
    all_loss = []
    for batch_output, batch_input in zip(epoch_output, epoch_input):
        all_prediction.append(batch_output['prediction'].detach().numpy())
        all_loss.append(batch_output['loss_batch'])
        all_labels.append(batch_input['labels'].numpy())
    all_prediction = np.vstack(all_prediction)
    all_labels = np.hstack(all_labels)
    accuracy = calculate_accuracy_classification(predictions=all_prediction, labels=all_labels)
    return accuracy

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



        # save all the batch stuff for further analysis
        output['loss_batch'] = loss.item()
        track_output.append(output)

    # Calculate all per-epoch metric
    accuracy = train_parameters.per_epoch_metric(track_output,train_parameters.iterator)
    return accuracy


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



