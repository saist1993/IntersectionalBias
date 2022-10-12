import math
import torch
from tqdm.auto import tqdm
from numpy.random import beta
import torch.nn.functional as F
from collections import Counter
from .common_functionality import *
from .mixup_training_loop import \
    train_only_mixup_with_abstract_group,\
    train_only_tilted_erm_with_abstract_group, \
    train_only_tilted_erm_with_mixup_augmentation,\
    train_only_mixup_based_on_distance,\
    train_with_mixup_only_one_group_based_distance, \
    train_only_mixup_based_on_distance_and_augmentation

def train_only_mixup(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    mixup_rg = train_tilted_params.other_params['mixup_rg']

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s_group_0, s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 2, replace=False)
        # s = F.gumbel_softmax(global_weight, tau=1/10, hard=True).nonzero()[0][0].item()

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
            # What we want is y=0,g=g1 and y=1,g=g1
            # here items_group_0 say with batch 500 -> first 250 are 0 label and next (last) 250 are 0 label
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

        composite_items = {
            'input': torch.vstack([items_group_0['input'], items_group_1['input']]),
            'labels': torch.hstack([items_group_0['labels'], items_group_1['labels']]),
            'aux': torch.vstack([items_group_0['aux'], items_group_1['aux']]),
            'aux_flattened': torch.hstack([items_group_0['aux_flattened'], items_group_1['aux_flattened']])
        }

        optimizer.zero_grad()
        output = model(composite_items)
        loss = criterion(output['prediction'], composite_items['labels'])



        # Mix up
        #
        # if abs(sum(items_group_1['aux'][0]) - sum(items_group_0['aux'][0])) > 1:
        #     alpha = 1.0
        #     gamma = beta(alpha, alpha)
        # else:
        #     alpha = 1.0
        #     gamma = beta(alpha, alpha)
        alpha = 1.0
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
            if train_tilted_params.other_params['method'] == 'only_mixup_with_loss_group':
                loss_reg = torch.abs(E_grad)/torch.mean(loss[len(items_group_0['input']):])
            else:
                loss_reg = torch.abs(E_grad)

        elif train_tilted_params.fairness_function == 'equal_odds' or \
            train_tilted_params.fairness_function == 'equal_opportunity':
            split_index = int(train_tilted_params.other_params['batch_size']/2)
            if train_tilted_params.fairness_function == 'equal_odds':
                gold_labels = [0,1]
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



                batch_x_mix = items_group_0['input'][index_start:index_end] * gamma + items_group_1['input'][index_start:index_end] * (1 - gamma)
                batch_x_mix = batch_x_mix.requires_grad_(True)
                output_mixup = model({'input':batch_x_mix})
                gradx = torch.autograd.grad(output_mixup['prediction'].sum(), batch_x_mix, create_graph=True)[0] # may be .sum()

                batch_x_d = items_group_1['input'][index_start:index_end] - items_group_0['input'][index_start:index_end]
                grad_inn = (gradx * batch_x_d).sum(1)
                E_grad = grad_inn.mean(0)
                if train_tilted_params.other_params['method'] == 'only_mixup_with_loss_group':
                    loss_reg = loss_reg +  torch.abs(E_grad) / torch.mean(loss[index_start:index_end])
                else:
                    loss_reg = loss_reg + torch.abs(E_grad)
                # loss_reg = loss_reg + torch.abs(E_grad)

        else:
            raise NotImplementedError

        loss = torch.mean(loss)
        loss = loss + mixup_rg*loss_reg
        loss.backward()
        optimizer.step()


        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(composite_items)

    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)
    return epoch_metric_tracker, loss, global_weight, global_loss


def train_with_mixup(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    mixup_rg = train_tilted_params.other_params['mixup_rg']


    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s_group_0, s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 2, p=global_weight, replace=False)
        # s_group_0 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight, replace=False)[0]
        break_flag = True
        while break_flag:
            temp_global_weight = np.reciprocal(global_weight)
            temp_global_weight = temp_global_weight / np.linalg.norm(temp_global_weight, 1)# Invert all weights
            s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=temp_global_weight, replace=False)[0]

            if s_group_1 != s_group_0:
                break_flag = False
        # s = F.gumbel_softmax(global_weight, tau=1/10, hard=True).nonzero()[0][0].item()

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
            # What we want is y=0,g=g1 and y=1,g=g1
            # here items_group_0 say with batch 500 -> first 250 are 0 label and next (last) 250 are 0 label
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

        composite_items = {
            'input': torch.vstack([items_group_0['input'], items_group_1['input']]),
            'labels': torch.hstack([items_group_0['labels'], items_group_1['labels']]),
            'aux': torch.vstack([items_group_0['aux'], items_group_1['aux']]),
            'aux_flattened': torch.hstack([items_group_0['aux_flattened'], items_group_1['aux_flattened']])
        }

        optimizer.zero_grad()
        output = model(composite_items)
        loss = criterion(output['prediction'], composite_items['labels'])
        loss_without_backward = torch.clone(loss).detach()
        loss_without_backward_group_0 = torch.mean(loss_without_backward[:len(items_group_0['input'])])
        loss_without_backward_group_1 = torch.mean(loss_without_backward[len(items_group_0['input']):])
        loss = torch.mean(loss)



        global_loss[s_group_0] =  0.2 * torch.exp(tilt_t*loss_without_backward_group_0) + 0.8 * global_loss[s_group_0]
        global_loss[s_group_1] =  0.2 * torch.exp(tilt_t*loss_without_backward_group_1) + 0.8 * global_loss[s_group_1]

        # weights = torch.exp(tao*loss_without_backward - tao*global_loss[s])
        global_weight = global_loss / torch.sum(global_loss)
        # global_weight = global_loss
        # loss = torch.mean(weights*loss)


        # fair mixup now
        alpha = 1
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


        loss = loss + mixup_rg*loss_reg



        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(composite_items)



    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)



    return epoch_metric_tracker, loss, global_weight, global_loss


def train_with_mixup_only_one_group(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    mixup_rg = train_tilted_params.other_params['mixup_rg']


    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s_group_0, s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 2, p=global_weight, replace=False)
        # s_group_0 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight, replace=False)[0]
        break_flag = True
        while break_flag:
            temp_global_weight = np.reciprocal(global_weight)
            temp_global_weight = temp_global_weight / np.linalg.norm(temp_global_weight, 1)# Invert all weights
            s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 1, p=temp_global_weight, replace=False)[0]

            if s_group_1 != s_group_0:
                break_flag = False
        # s = F.gumbel_softmax(global_weight, tau=1/10, hard=True).nonzero()[0][0].item()

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
            # What we want is y=0,g=g1 and y=1,g=g1
            # here items_group_0 say with batch 500 -> first 250 are 0 label and next (last) 250 are 0 label
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

        # composite_items = {
        #     'input': torch.vstack([items_group_0['input'], items_group_1['input']]),
        #     'labels': torch.hstack([items_group_0['labels'], items_group_1['labels']]),
        #     'aux': torch.vstack([items_group_0['aux'], items_group_1['aux']]),
        #     'aux_flattened': torch.hstack([items_group_0['aux_flattened'], items_group_1['aux_flattened']])
        # }

        optimizer.zero_grad()
        output = model(items_group_0)
        loss = criterion(output['prediction'], items_group_0['labels'])
        loss_without_backward = torch.mean(torch.clone(loss).detach())
        # loss_without_backward_group_0 = torch.mean(loss_without_backward[:len(items_group_0['input'])])
        # loss_without_backward_group_1 = torch.mean(loss_without_backward[len(items_group_0['input']):])
        loss = torch.mean(loss)



        global_loss[s_group_0] =  0.2 * torch.exp(tilt_t*loss_without_backward) + 0.8 * global_loss[s_group_0]
        # global_loss[s_group_1] =  0.2 * torch.exp(tilt_t*loss_without_backward_group_1) + 0.8 * global_loss[s_group_1]

        # weights = torch.exp(tao*loss_without_backward - tao*global_loss[s])
        global_weight = global_loss / torch.sum(global_loss)
        # global_weight = global_loss
        # loss = torch.mean(weights*loss)


        # fair mixup now
        alpha = 1
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


        loss = loss + mixup_rg*loss_reg



        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items_group_0)



    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)



    return epoch_metric_tracker, loss, global_weight, global_loss


def train_tilted_erm_with_fairness_loss(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']
    mixup_rg = train_tilted_params.other_params['mixup_rg']


    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s_group_0, s_group_1 = np.random.choice(train_tilted_params.other_params['groups'], 2, p=global_weight, replace=False)
        # s = F.gumbel_softmax(global_weight, tau=1/10, hard=True).nonzero()[0][0].item()

        if train_tilted_params.fairness_function == 'equal_opportunity':
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

        else:
            raise NotImplementedError

        for key in items_group_0.keys():
            items_group_0[key] = items_group_0[key].to(train_tilted_params.device)

        for key in items_group_1.keys():
            items_group_1[key] = items_group_1[key].to(train_tilted_params.device)

        composite_items = {
            'input': torch.vstack([items_group_0['input'], items_group_1['input']]),
            'labels': torch.hstack([items_group_0['labels'], items_group_1['labels']]),
            'aux': torch.vstack([items_group_0['aux'], items_group_1['aux']]),
            'aux_flattened': torch.hstack([items_group_0['aux_flattened'], items_group_1['aux_flattened']])
        }

        optimizer.zero_grad()
        output = model(composite_items)
        loss = criterion(output['prediction'], composite_items['labels'])
        loss_without_backward_group_0 = torch.mean(loss[:len(items_group_0['input'])])
        loss_without_backward_group_1 = torch.mean(loss[len(items_group_0['input']):])
        loss = torch.mean(loss)




        global_loss[s_group_0] =  0.2 * torch.exp(tilt_t*torch.clone(loss_without_backward_group_0).detach()) + 0.8 * global_loss[s_group_0]
        global_loss[s_group_1] =  0.2 * torch.exp(tilt_t*torch.clone(loss_without_backward_group_1).detach()) + 0.8 * global_loss[s_group_1]

        # weights = torch.exp(tao*loss_without_backward - tao*global_loss[s])
        global_weight = global_loss / torch.sum(global_loss)
        # global_weight = global_loss
        # loss = torch.mean(weights*loss)


        # fair mixup now



        loss = loss + mixup_rg*abs(loss_without_backward_group_0 - loss_without_backward_group_1)



        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(composite_items)



    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)



    return epoch_metric_tracker, loss, global_weight, global_loss


def train_only_tilted_erm(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        # s = F.gumbel_softmax(global_weight, tau=1/10, hard=True).nonzero()[0][0].item()

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
        loss = torch.mean(criterion(output['prediction'], items['labels']))
        loss_without_backward = torch.clone(loss).detach()

        # tilt the loss
        # loss_r_b = torch.log(torch.mean(torch.exp(tao * loss_without_backward)))/tao


        global_loss[s] =  0.2 * torch.exp(tilt_t*loss_without_backward) + 0.8 * global_loss[s]

        # weights = torch.exp(tao*loss_without_backward - tao*global_loss[s])
        global_weight = global_loss / torch.sum(global_loss)
        # global_weight = global_loss
        # loss = torch.mean(weights*loss)
        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)



    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)


    return epoch_metric_tracker, loss, global_weight, global_loss



def train_only_tilted_dro(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

    for i in tqdm(range(train_tilted_params.other_params['number_of_iterations'])):
        s = np.random.choice(train_tilted_params.other_params['groups'], 1, p=global_weight)[0]
        # s = F.gumbel_softmax(global_weight, tau=1/10, hard=True).nonzero()[0][0].item()

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
        loss = torch.mean(criterion(output['prediction'], items['labels']))
        loss_without_backward = torch.clone(loss).detach()

        # tilt the loss
        # loss_r_b = torch.log(torch.mean(torch.exp(tao * loss_without_backward)))/tao


        global_loss[s] =   global_loss[s]*torch.exp(tilt_t*loss_without_backward)

        # weights = torch.exp(tao*loss_without_backward - tao*global_loss[s])
        global_weight = global_loss / torch.sum(global_loss)
        # global_weight = global_loss
        # loss = torch.mean(weights*loss)
        loss = loss*global_weight[s]
        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)



    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)


    return epoch_metric_tracker, loss, global_weight, global_loss


def train_weighted_sample_erm(train_tilted_params:TrainParameters):

    global_weight = train_tilted_params.other_params['global_weight']
    global_loss = train_tilted_params.other_params['global_loss']
    tilt_t = train_tilted_params.other_params['titled_t']

    model, optimizer, device, criterion = \
        train_tilted_params.model, train_tilted_params.optimizer, train_tilted_params.device, train_tilted_params.criterion
    model.train()
    track_output = []
    track_input = []

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
        loss = torch.mean(criterion(output['prediction'], items['labels']))
        loss.backward()
        optimizer.step()

        output['loss_batch'] = loss.item()
        track_output.append(output)
        track_input.append(items)



    epoch_metric_tracker, loss = train_tilted_params.per_epoch_metric(track_output,
                                                                   track_input,
                                                                   train_tilted_params.fairness_function)


    return epoch_metric_tracker, loss, global_weight, global_loss



def test(train_parameters: TrainParameters):
    """Trains the model for one epoch"""
    model, optimizer, device, criterion, mode = \
        train_parameters.model, train_parameters.optimizer, train_parameters.device, train_parameters.criterion, train_parameters.mode


    model.eval()
    track_output = []


    for items in tqdm(train_parameters.iterator):

        # Change the device of all the tensors!
        for key in items.keys():
            items[key] = items[key].to(device)


        with torch.no_grad():
            output = model(items)
            loss = torch.mean(criterion(output['prediction'], items['labels']))  # As reduction is None.


        # Save all batch stuff for further analysis
        output['loss_batch'] = loss.item()
        track_output.append(output)

    # Calculate all per-epoch metric by sending outputs and the inputs
    epoch_metric_tracker, loss = train_parameters.per_epoch_metric(track_output,
                                                                   train_parameters.iterator,
                                                                   train_parameters.fairness_function)
    return epoch_metric_tracker, loss


def sample_batch_sen_idx(all_input, all_label, all_aux, all_aux_flatten, batch_size, s):
    """
        This will sample batch size number of elements from input with given s!
    """
    relevant_index = np.random.choice(np.where(all_aux_flatten==s)[0], size=batch_size, replace=True).tolist()
    # THIS IS DIFFERENT. IN ORIGINAL VERSION IT IS REPLACEMENT TRUE
    batch_input = {
        'labels': torch.LongTensor(all_label[relevant_index]),
        'input': torch.FloatTensor(all_input[relevant_index]),
        'aux': torch.LongTensor(all_aux[relevant_index]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[relevant_index])
    }

    return batch_input



def sample_batch_sen_idx_with_y(all_input, all_label, all_aux, all_aux_flatten, batch_size, s):
    """
        This will sample batch size number of elements from input with given s!
    """
    index_s_0 = np.where(np.logical_and(all_aux_flatten==s, all_label==0) == True)[0]
    index_s_1 = np.where(np.logical_and(all_aux_flatten==s, all_label==1) == True)[0]
    relevant_index = np.random.choice(index_s_0, size=int(batch_size/2), replace=True).tolist()
    relevant_index = relevant_index + np.random.choice(index_s_1,size=int(batch_size/2), replace=True).tolist()

    # THIS IS DIFFERENT. IN ORIGINAL VERSION IT IS REPLACEMENT TRUE
    batch_input = {
        'labels': torch.LongTensor(all_label[relevant_index]),
        'input': torch.FloatTensor(all_input[relevant_index]),
        'aux': torch.LongTensor(all_aux[relevant_index]),
        'aux_flattened': torch.LongTensor(all_aux_flatten[relevant_index])
    }

    return batch_input



def training_loop(training_loop_parameters: TrainingLoopParameters):
    '''

    :param training_loop_parameters:
    :return:
    This training loop would look different than all others. As this requires access to dataset based on s

    input_data = {
            'labels': labels,
            'input': encoded_input,
            'lengths': lengths,
            'aux': aux,
            'aux_flattened': aux_flattened
        }


    '''
    logger = logging.getLogger(training_loop_parameters.unique_id_for_run)
    output = {}
    all_train_eps_metrics = []
    all_test_eps_metrics = []
    all_valid_eps_metrics = []
    best_test_accuracy = 0.0
    best_eopp = 1.0

    training_loop_type = training_loop_parameters.other_params['method']
    # Housekeeping because we cannot use a simple iter which is returned by the dataset_parser!
    _labels, _input, _lengths, _aux, _aux_flattened = [], [], [], [], []
    for items in (training_loop_parameters.iterators[0]['train_iterator']):
        _labels.append(items['labels'])
        _input.append(items['input'])
        _lengths.append(items['lengths'])
        _aux.append(items['aux'])
        _aux_flattened.append(items['aux_flattened'])

    all_label = np.hstack(_labels)
    all_aux = np.vstack(_aux)
    all_aux_flatten = np.hstack(_aux_flattened)
    all_input = np.vstack(_input)
    size_of_training_dataset = len(all_label)

    total_no_groups = len(np.unique(all_aux_flatten))

    counts = Counter(all_aux_flatten)
    # size_of_each_group = [counts[i] for i in range(total_no_groups)]

    # Now comes the traing loop and evaluation loop
    # global_weight = torch.tensor(np.full(total_no_groups, 1.0/total_no_groups))
    # global_loss = torch.tensor(np.full(total_no_groups, 1.0/total_no_groups))

    if training_loop_type == 'weighted_sample_erm' or training_loop_type == 'only_titled_erm_with_weights':
        size_of_each_group = [counts[i] for i in range(total_no_groups)]
        weights = np.asarray([1.0/i for i in size_of_each_group])
        global_weight = torch.tensor(weights / np.linalg.norm(weights, 1))
        global_loss = torch.tensor(weights / np.linalg.norm(weights, 1))
    else:
        weights = np.asarray([1/total_no_groups for i in range(total_no_groups)])
        global_weight = torch.tensor(weights/np.linalg.norm(weights, 1))
        global_loss = torch.tensor(weights/np.linalg.norm(weights, 1))

    # global_weight = global_weight/ torch.norm(global_weight, 1)
    # global_loss = global_weight/ torch.norm(global_loss, 1)
    groups = [i for i in range(total_no_groups)]

    # models = []


    for ep in range(training_loop_parameters.n_epochs):

        logger.info("start of epoch block  ")

        training_loop_parameters.other_params['number_of_iterations'] = int(size_of_training_dataset/training_loop_parameters.other_params['batch_size'])
        training_loop_parameters.other_params['global_weight'] = global_weight
        training_loop_parameters.other_params['global_loss'] = global_loss
        training_loop_parameters.other_params['groups'] = groups
        training_loop_parameters.other_params['all_label'] = all_label
        training_loop_parameters.other_params['all_aux'] = all_aux
        training_loop_parameters.other_params['all_aux_flatten'] = all_aux_flatten
        training_loop_parameters.other_params['all_input'] = all_input
        training_loop_parameters.other_params['valid_iterator'] = training_loop_parameters.iterators[0]['valid_iterator']

        train_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['train_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='train',
            fairness_function=training_loop_parameters.fairness_function)

        if training_loop_type == 'tilted_erm_with_mixup':
            train_epoch_metric, loss, global_weight, global_loss = train_with_mixup(train_parameters)
        elif training_loop_type == 'tilted_erm_with_mixup_only_one_group':
            train_epoch_metric, loss, global_weight, global_loss = train_with_mixup_only_one_group(train_parameters)
        elif training_loop_type == 'only_mixup' or training_loop_type == 'only_mixup_with_loss_group':
            train_epoch_metric, loss, global_weight, global_loss = train_only_mixup(train_parameters)
        elif training_loop_type == 'only_titled_erm' or training_loop_type == 'only_titled_erm_with_weights':
            train_epoch_metric, loss, global_weight, global_loss = train_only_tilted_erm(train_parameters)
        elif training_loop_type == 'tilted_erm_with_fairness_loss':
            train_epoch_metric, loss, global_weight, global_loss = train_tilted_erm_with_fairness_loss(train_parameters)
        elif training_loop_type == 'only_mixup_with_abstract_group':
            train_epoch_metric, loss, global_weight, global_loss = train_only_mixup_with_abstract_group(train_parameters)
        elif training_loop_type == 'weighted_sample_erm':
            train_epoch_metric, loss, global_weight, global_loss = train_weighted_sample_erm(
                train_parameters)
        elif training_loop_type == 'only_tilted_erm_with_abstract_group':
            train_epoch_metric, loss, global_weight, global_loss = train_only_tilted_erm_with_abstract_group(
                train_parameters)
        elif training_loop_type == 'tilted_erm_with_mixup_augmentation':
            train_epoch_metric, loss, global_weight, global_loss = train_only_tilted_erm_with_mixup_augmentation(
                train_parameters)
        elif training_loop_type == 'only_mixup_based_on_distance':
            train_epoch_metric, loss, global_weight, global_loss = train_only_mixup_based_on_distance(
                train_parameters)
        elif training_loop_type == 'tilted_erm_with_mixup_based_on_distance':
            train_epoch_metric, loss, global_weight, global_loss = train_with_mixup_only_one_group_based_distance(
                train_parameters)
        elif training_loop_type == 'only_mixup_based_on_distance_and_augmentation':
            train_epoch_metric, loss, global_weight, global_loss = train_only_mixup_based_on_distance_and_augmentation(train_parameters)
        elif training_loop_type == 'only_tilted_dro':
            train_epoch_metric, loss, global_weight, global_loss = train_only_tilted_dro(train_parameters)
        else:
            raise NotImplementedError


        log_epoch_metric(logger, start_message='train', epoch_metric=train_epoch_metric, epoch_number=ep, loss=loss)
        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=train_epoch_metric, loss=loss, train=True)
        all_train_eps_metrics.append(train_epoch_metric)

        # Valid split
        valid_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['valid_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='evaluate',
            fairness_function=training_loop_parameters.fairness_function)

        valid_epoch_metric, loss = test(valid_parameters)
        log_epoch_metric(logger, start_message='valid', epoch_metric=valid_epoch_metric, epoch_number=ep, loss=loss)

        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=valid_epoch_metric, loss=loss, train=True)
        all_valid_eps_metrics.append(train_epoch_metric)

        # test split
        test_parameters = TrainParameters(
            model=training_loop_parameters.model,
            iterator=training_loop_parameters.iterators[0]['test_iterator'],
            optimizer=training_loop_parameters.optimizer,
            criterion=training_loop_parameters.criterion,
            device=training_loop_parameters.device,
            other_params=training_loop_parameters.other_params,
            per_epoch_metric=per_epoch_metric,
            mode='evaluate',
            fairness_function=training_loop_parameters.fairness_function)

        test_epoch_metric, loss = test(test_parameters)
        if training_loop_parameters.use_wandb:
            log_and_plot_data(epoch_metric=test_epoch_metric, loss=loss, train=False)
        log_epoch_metric(logger, start_message='test', epoch_metric=test_epoch_metric, epoch_number=ep, loss=loss)
        all_test_eps_metrics.append(test_epoch_metric)

        logger.info("end of epoch block")

        # models.append(copy.deepcopy(training_loop_parameters.model))

    output['all_train_eps_metric'] = all_train_eps_metrics
    output['all_test_eps_metric'] = all_test_eps_metrics
    output['all_valid_eps_metric'] = all_valid_eps_metrics
    # index = find_best_model(output, training_loop_parameters.fairness_function)
    # output['best_model_index'] = index
    # method = training_loop_parameters.other_params['method']
    # dataset_name = training_loop_parameters.other_params['dataset_name']
    # seed = training_loop_parameters.other_params['seed']
    # _dir = f'../saved_models/{dataset_name}/{method}/{seed}'
    # Path(_dir).mkdir(parents=True, exist_ok=True)
    # if training_loop_parameters.save_model_as != None:
    #     torch.save(models[index].state_dict(),
    #                f'{_dir}/{training_loop_parameters.fairness_function}_{training_loop_parameters.save_model_as}.pt')

    return output
