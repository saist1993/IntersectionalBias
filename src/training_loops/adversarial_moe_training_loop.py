import torch
import wandb
import numpy as np
import torch.nn as nn
from utils import misc
from tqdm.auto import tqdm
from .common_functionality import *
from metrics import calculate_epoch_metric, fairness_utils



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
            if train_parameters.other_params['fairness_lambda'] != 0.0:
                fairness_loss = \
                    get_fairness_loss \
                        (train_parameters.fairness_function, loss, output['prediction'], items['aux'],
                         items['labels'],
                         all_independent_group_patterns)
                if fairness_loss:
                    loss = torch.mean(loss) + train_parameters.other_params['fairness_lambda'] * fairness_loss
                else:
                    loss = torch.mean(loss)
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
    epoch_metric_tracker, loss = train_parameters.per_epoch_metric(track_output,
                                                                   train_parameters.iterator,
                                                                   train_parameters.fairness_function,
                                                                   attribute_id)
    return epoch_metric_tracker, loss


def training_loop(training_loop_parameters: TrainingLoopParameters):
    output = training_loop_common(training_loop_parameters, train)
    return output
