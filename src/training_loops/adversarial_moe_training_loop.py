import torch
import wandb
import numpy as np
import torch.nn as nn
from utils import misc
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Dict, Callable
from itertools import combinations
from metrics import calculate_epoch_metric, fairness_utils







# configuring matplot lib


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
    n_epochs: int
    model: nn.Module
    iterators: Dict
    optimizer: torch.optim
    criterion: Callable
    device: torch.device
    save_model_as: str
    use_wandb: bool
    other_params: Dict


def per_epoch_metric(epoch_output, epoch_input, attribute_id=None):
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
    if type(epoch_output[0]['adv_outputs']) is  list:
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
            all_adversarial_output.append(batch_output['adv_outputs'].detach().numpy())

        # all_s_prediction(batch_output[''])

    if not flag:
        # this is a adversarial group
        for j in range(len(epoch_output[0]['adv_outputs'])):
            all_adversarial_output.append(torch.vstack([i['adv_outputs'][j] for i in epoch_output]).argmax(1).detach().numpy())
    else:
        all_adversarial_output = np.vstack(all_adversarial_output).argmax(1)
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
        'fairness_mode': ['equal_opportunity'],
        'no_fairness': False,
        'adversarial_output': all_adversarial_output,
        'aux_flattened': all_s_flatten
    }

    epoch_metric = calculate_epoch_metric.CalculateEpochMetric(all_prediction, all_label, all_s, other_meta_data).run()
    # epoch_metric = None
    return epoch_metric, np.mean(all_loss)


def get_fairness_loss_equal_opportunity(loss, preds, aux, label, all_patterns):
    losses = []
    label_mask = label == 1
    for pattern in all_patterns:
        aux_mask = torch.einsum("ij->i", torch.eq(aux, pattern)) > aux.shape[1] - 1
        final_mask = torch.logical_and(label_mask, aux_mask)
        _loss = loss[final_mask]
        if len(_loss) > 0:
            losses.append(torch.mean(_loss))
    final_loss = []
    for l1, l2 in combinations(losses, 2):
        final_loss.append(abs(l1-l2))
    if len(losses) == 0:
        return None

    return torch.stack(losses).sum()

def train(train_parameters: TrainParameters):
    """Trains the model for one epoch"""
    model, optimizer, device, criterion, mode = train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion, train_parameters.mode

    adversarial_method = train_parameters.other_params['method']
    adversarial_lambda = train_parameters.other_params['adversarial_lambda']
    attribute_id = train_parameters.other_params['attribute_id']

    # Method to find all independent group patterns. This will be used in fairness loss.
    all_s = []
    for items in train_parameters.iterator:
        all_s.append(items['aux'].numpy())
    all_s = np.vstack(all_s)
    all_independent_group_patterns = fairness_utils.create_all_possible_groups \
        (attributes=[list(np.unique(all_s[:, i])) for i in range(all_s.shape[1])])
    all_independent_group_patterns = [torch.tensor(i) for i in all_independent_group_patterns if 'x' not in i]


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
            loss = criterion(output['prediction'], items['labels'])  # As reduction is None.
            fairness_loss = \
                get_fairness_loss_equal_opportunity\
                    (loss, output['prediction'], items['aux'], items['labels'], all_independent_group_patterns)
            if fairness_loss:
                loss = torch.mean(loss) + 0.05*fairness_loss
            else:
                 loss = torch.mean(loss)

            for i in range(len(output['adv_outputs'])):
                loss_interm = torch.mean(criterion(output['adv_outputs'][i], items['aux'][:,i]))
                loss = loss + adversarial_lambda * loss_interm

            loss.backward()
            optimizer.step()
        elif mode == 'evaluate':
            with torch.no_grad():
                output = model(items)
                loss = torch.mean(criterion(output['prediction'], items['labels']))  # As reduction is None.
                if adversarial_method == 'adversarial_single':
                    loss_aux = torch.mean(criterion(output['adv_outputs'][0], items['aux_flattened']))
                    loss = loss + adversarial_lambda * loss_aux  # make this parameterized!
                    output['adv_outputs'] = output['adv_outputs'][0]  # makes it easier for further changes
                elif adversarial_method == 'adversarial_group':
                    if attribute_id is not None:
                        loss_aux = torch.mean(criterion(output['adv_outputs'][0], items['aux'][:, attribute_id]))
                    else:
                        loss_aux = torch.mean(
                        torch.tensor([torch.mean(criterion(output['adv_outputs'][i], items['aux'][:,i])) for i in
                         range(len(output['adv_outputs']))]))
                    loss = loss + adversarial_lambda * loss_aux
        else:
            raise misc.CustomError("only supports train and evaluate")

        # Save all batch stuff for further analysis
        output['loss_batch'] = loss.item()
        track_output.append(output)

    # Calculate all per-epoch metric by sending outputs and the inputs
    epoch_metric_tracker, loss = train_parameters.per_epoch_metric(track_output, train_parameters.iterator, attribute_id)
    return epoch_metric_tracker, loss


def log_and_plot_data(epoch_metric, loss, train=True):
    if train:
        suffix = "train_"
    else:
        suffix = "test_"

    wandb.log({suffix + "accuracy": epoch_metric.accuracy,
               suffix + "balanced_accuracy": epoch_metric.balanced_accuracy,
               suffix + "loss": loss})

    # generate the matplotlib graph!


def training_loop(training_loop_parameters: TrainingLoopParameters):
    output = {}
    all_train_eps_metrics = []
    all_test_eps_metrics = []
    best_eopp = 1.0





    best_test_accuracy = 0.0
    for _ in range(training_loop_parameters.n_epochs):
        train_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['train_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='train')

        train_epoch_metric, loss = train(train_parameters)
        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=train_epoch_metric, loss=loss, train=True)
        all_train_eps_metrics.append(train_epoch_metric.eps_fairness)

        test_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['test_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='evaluate')

        test_epoch_metric, loss = train(test_parameters)
        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=test_epoch_metric, loss=loss, train=False)
        all_test_eps_metrics.append(test_epoch_metric.eps_fairness)

        print(f"train epoch metric is {train_epoch_metric}")
        print(f"test epoch metric is {test_epoch_metric}")

        equal_opp = test_epoch_metric.eps_fairness['equal_opportunity'].intersectional_bootstrap[0]
        # if best_test_accuracy < test_epoch_metric.accuracy:
        #     best_test_accuracy = test_epoch_metric.accuracy
        #     if training_loop_parameters.save_model_as:
        #         print("model saved")
        #         torch.save(training_loop_parameters.model.state_dict(), training_loop_parameters.save_model_as + ".pt")

        if test_epoch_metric.accuracy > 0.81:
            if equal_opp < best_eopp:
                best_eopp = equal_opp
                if training_loop_parameters.save_model_as:
                    print("model saved")
                    torch.save(training_loop_parameters.model.state_dict(), training_loop_parameters.save_model_as + ".pt")


    output['all_train_eps_metric'] = all_train_eps_metrics
    output['all_test_eps_metric'] = all_test_eps_metrics
    output['trained_model_last_epoch'] = training_loop_parameters.model


    return output